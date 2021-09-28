/*
 * (c) Copyright 2019-2021 Xilinx, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
#ifndef _XFCOMPRESSION_ZLIB_COMPRESS_DETAILS_HPP_
#define _XFCOMPRESSION_ZLIB_COMPRESS_DETAILS_HPP_

/**
 * @file zlib_compress_details.hpp
 * @brief Header for modules used in ZLIB compression kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include <ap_int.h>
#include <assert.h>
#include <stdint.h>

#include "compress_utils.hpp"
#include "zlib_specs.hpp"
#include "lz_optional.hpp"
#include "lz_compress.hpp"
#include "huffman_treegen.hpp"
#include "huffman_encoder.hpp"
#include "mm2s.hpp"
#include "s2mm.hpp"
#include "stream_downsizer.hpp"
#include "stream_upsizer.hpp"
#include "xf_security/adler32.hpp"
#include "xf_security/crc32.hpp"

namespace xf {
namespace compression {
namespace details {

template <int SLAVES>
void streamEosDistributor(hls::stream<bool>& inStream, hls::stream<bool> outStream[SLAVES]) {
    do {
        bool i = inStream.read();
        for (int n = 0; n < SLAVES; n++) outStream[n] << i;
        if (i == 1) break;
    } while (1);
}

template <int SLAVES>
void streamSizeDistributor(hls::stream<uint32_t>& inStream, hls::stream<uint32_t> outStream[SLAVES]) {
    do {
        uint32_t i = inStream.read();
        for (int n = 0; n < SLAVES; n++) outStream[n] << i;
        if (i == 0) break;
    } while (1);
}

template <int DWIDTH = 64, int SIZE_DWIDTH = 4, int PARALLEL_BYTES = 8, int STRATEGY = 0>
void dataDuplicator(hls::stream<IntVectorStream_dt<8, PARALLEL_BYTES> >& inStream,
                    hls::stream<ap_uint<32> >& checksumInitStream,
                    hls::stream<ap_uint<PARALLEL_BYTES * 8> >& checksumOutStream,
                    hls::stream<ap_uint<5> >& checksumSizeStream,
                    hls::stream<ap_uint<DWIDTH + SIZE_DWIDTH> >& coreStream) {
    constexpr int incr = DWIDTH / 8;

    // Checksum initial data
    if (STRATEGY == 0) {
        checksumInitStream << ~0;
    } else {
        checksumInitStream << 0;
    }

duplicator:
    while (1) {
        IntVectorStream_dt<8, PARALLEL_BYTES> tmpVal = inStream.read();
        // Last will be high if strobe is 0
        bool last = (tmpVal.strobe == 0);
        ap_uint<DWIDTH + SIZE_DWIDTH> outVal = 0;
        // First SIZE_DWIDTH bits will be no. of valid bytes
        outVal.range(SIZE_DWIDTH - 1, 0) = tmpVal.strobe;
        // Checksum requires the parallel valid bytes
        checksumSizeStream << (ap_uint<5>)tmpVal.strobe;
        // Data parallel write
        for (auto i = SIZE_DWIDTH, j = 0; i < (DWIDTH + SIZE_DWIDTH); i += incr) {
#pragma HLS UNROLL
            outVal.range(i + 7, i) = tmpVal.data[j++];
        }
        // Core Data Stream
        coreStream << outVal;
        if (last) break;
        // Checksum Data Stream
        checksumOutStream << outVal.range(DWIDTH + SIZE_DWIDTH - 1, SIZE_DWIDTH);
    }
}

template <int DWIDTH = 64, int PARALLEL_BYTES = 8>
void dataDuplicator(hls::stream<ap_uint<DWIDTH> >& inStream,
                    hls::stream<uint32_t>& inSizeStream,
                    hls::stream<ap_uint<PARALLEL_BYTES * 8> >& checksumOutStream,
                    hls::stream<ap_uint<5> >& checksumSizeStream,
                    hls::stream<ap_uint<DWIDTH> >& coreStream,
                    hls::stream<uint32_t>& coreSizeStream) {
    constexpr int c_parallelByte = DWIDTH / 8;
    uint32_t inputSize = inSizeStream.read();

    coreSizeStream << inputSize;

    uint32_t inSize = (inputSize - 1) / c_parallelByte + 1;
    bool inSizeMod = (inputSize % c_parallelByte == 0);

duplicator:
    for (uint32_t i = 0; i < inSize; i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<DWIDTH> inValue = inStream.read();
        auto c_size = (i == inSize - 1) && !inSizeMod ? (inputSize % c_parallelByte) : c_parallelByte;
        checksumSizeStream << c_size;
        checksumOutStream << inValue;
        coreStream << inValue;
    }
    checksumSizeStream << 0;
}

template <int FREQ_DWIDTH = 32, int DWIDTH = 64, int NUM_BLOCKS = 8, int BLOCK_SIZE = 32768, int MIN_BLCK_SIZE = 64>
void multicoreDistributor(hls::stream<ap_uint<DWIDTH> >& inStream,
                          hls::stream<uint32_t>& inSizeStream,
                          hls::stream<ap_uint<DWIDTH> >& strdStream,
                          hls::stream<ap_uint<16> >& strdSizeStream,
                          hls::stream<ap_uint<DWIDTH> > distStream[NUM_BLOCKS],
                          hls::stream<ap_uint<FREQ_DWIDTH> > distSizeStream[NUM_BLOCKS]) {
    constexpr int incr = DWIDTH / 8;
    ap_uint<4> core = 0;
    uint32_t readSize = 0;
    uint32_t inputSize = inSizeStream.read();

    while (readSize < inputSize) {
        core %= NUM_BLOCKS;
        ap_uint<FREQ_DWIDTH> outputSize = ((readSize + BLOCK_SIZE) > inputSize) ? (inputSize - readSize) : BLOCK_SIZE;

        bool storedBlockFlag = (outputSize <= MIN_BLCK_SIZE);

        readSize += outputSize;
        if (!storedBlockFlag)
            distSizeStream[core] << outputSize;
        else
            strdSizeStream << outputSize;

    multicoreDistributor:
        for (ap_uint<FREQ_DWIDTH> j = 0; j < outputSize; j += incr) {
#pragma HLS PIPELINE II = 1
            ap_uint<DWIDTH> outData = inStream.read();
            if (storedBlockFlag)
                strdStream << outData;
            else
                distStream[core] << outData;
        }
        core++;
    }

    for (int i = 0; i < NUM_BLOCKS; i++) {
        distSizeStream[i] << 0;
    }
    strdSizeStream << 0;
}

template <class SIZE_DT = uint64_t,
          int DWIDTH = 64,
          int SIZE_DWIDTH = 4,
          int NUM_BLOCKS = 8,
          int BLOCK_SIZE = 32768,
          int MIN_BLCK_SIZE = 64,
          int STRATEGY = 0>
void multicoreDistributor(hls::stream<ap_uint<DWIDTH + SIZE_DWIDTH> >& inStream,
                          hls::stream<SIZE_DT>& fileSizeStream,
                          hls::stream<ap_uint<DWIDTH> >& strdStream,
                          hls::stream<ap_uint<16> >& strdSizeStream,
                          hls::stream<bool>& blckEosStream,
                          hls::stream<ap_uint<4> >& coreIdxStream,
                          hls::stream<ap_uint<DWIDTH + SIZE_DWIDTH> > distStream[NUM_BLOCKS]) {
    constexpr int c_incr = DWIDTH / 8;
    constexpr int c_factor = (MIN_BLCK_SIZE + c_incr) / c_incr;
    constexpr int c_storedDepth = 2 * c_factor;
    static ap_uint<4> core = 0;
    coreIdxStream << core;

    SIZE_DT readSize = 0;
    SIZE_DT writeSize = 0;
    ap_uint<16> strdCntr = 0;
    ap_uint<SIZE_DWIDTH> strobe = 0;
    bool last = false;

    hls::stream<ap_uint<DWIDTH + SIZE_DWIDTH> > storedBufferStream;
#pragma HLS STREAM variable = storedBufferStream depth = c_storedDepth

init_loop:
    for (uint8_t i = 0; i < c_factor && !last; i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<DWIDTH + SIZE_DWIDTH> tmpVal = inStream.read();
        strobe = tmpVal.range(SIZE_DWIDTH - 1, 0);
        last = (strobe == 0);
        readSize += strobe;
        storedBufferStream << tmpVal;
    }

distributor:
    while (!last) {
        core %= NUM_BLOCKS;
        blckEosStream << false;

        for (uint32_t i = 0; i < BLOCK_SIZE && !last; i += c_incr) {
#pragma HLS PIPELINE II = 1
            ap_uint<DWIDTH + SIZE_DWIDTH> tmpVal = inStream.read();
            ap_uint<SIZE_DWIDTH> strobe = tmpVal.range(SIZE_DWIDTH - 1, 0);
            readSize += strobe;
            last = (strobe == 0);
            ap_uint<DWIDTH + SIZE_DWIDTH> tmp = storedBufferStream.read();
            distStream[core] << tmp;
            strdCntr += tmp.range(SIZE_DWIDTH - 1, 0);
            writeSize += tmp.range(SIZE_DWIDTH - 1, 0);
            storedBufferStream << tmpVal;
        }

        if (!last) {
            distStream[core] << 0;
            strdCntr = 0;
            core++;
        }
    }

    blckEosStream << true;
    bool onlyOnce = true;
    bool endBlck = true;
    for (; writeSize != readSize && strdCntr != 0;) {
#pragma HLS PIPELINE II = 1
        ap_uint<SIZE_DWIDTH + DWIDTH> tmpVal = storedBufferStream.read();
        distStream[core] << tmpVal;
        strdCntr += tmpVal.range(SIZE_DWIDTH - 1, 0);
        writeSize += tmpVal.range(SIZE_DWIDTH - 1, 0);
        if (strdCntr == BLOCK_SIZE) {
            endBlck = false;
            strdCntr = 0;
        }
    }

    if (!endBlck) {
        if (readSize == writeSize)
            distStream[core] << storedBufferStream.read();
        else
            distStream[core] << 0;
        core++;
    } else if (endBlck && strdCntr > 0) {
        distStream[core] << storedBufferStream.read();
        core++;
    }

    for (; (writeSize != readSize && strdCntr == 0) || onlyOnce;) {
#pragma HLS PIPELINE II = 1
        if (onlyOnce) {
            strdSizeStream << (readSize - writeSize);
            onlyOnce = false;
            if (readSize == writeSize) break;
        }
        ap_uint<SIZE_DWIDTH + DWIDTH> tmpVal = storedBufferStream.read();
        strdStream << tmpVal.range(SIZE_DWIDTH + DWIDTH - 1, SIZE_DWIDTH);
        writeSize += tmpVal.range(SIZE_DWIDTH - 1, 0);
        if (readSize == writeSize) storedBufferStream.read();
    }

    // Total Input Size for GZIP only
    if (STRATEGY == 0) fileSizeStream << writeSize;

ip_terminate_data:
    for (uint8_t i = 0; i < NUM_BLOCKS; i++) {
        distStream[i] << 0;
    }
}

template <int DWIDTH = 64, int SIZE_DWIDTH = 4, int NUM_BLOCKS = 8, int BLOCK_SIZE = 32768, int MIN_BLCK_SIZE = 64>
void multicoreMerger(hls::stream<ap_uint<DWIDTH + SIZE_DWIDTH> > inStream[NUM_BLOCKS],
                     hls::stream<ap_uint<DWIDTH> >& strdStream,
                     hls::stream<ap_uint<16> >& strdSizeStream,
                     hls::stream<ap_uint<DWIDTH> >& outStream,
                     hls::stream<bool>& outStreamEos,
                     hls::stream<uint32_t>& outSizeStream) {
    constexpr int incr = DWIDTH / 8;
    uint32_t outSize = 0;
    ap_uint<NUM_BLOCKS> is_pending;
    ap_uint<2 * DWIDTH> inputWindow;
    uint32_t factor = DWIDTH / 8;
    uint32_t inputIdx = 0;

    for (uint8_t i = 0; i < NUM_BLOCKS; i++) {
#pragma HLS UNROLL
        is_pending.range(i, i) = 1;
    }

    while (is_pending) {
        for (int i = 0; i < NUM_BLOCKS; i++) {
            bool blockDone = false;
            bool not_first = false;
            for (; (blockDone == false) && (is_pending(i, i) == true);) {
#pragma HLS PIPELINE II = 1
                assert((inputIdx + factor) <= 2 * (DWIDTH / 8));

                ap_uint<DWIDTH + SIZE_DWIDTH> inVal = inStream[i].read();
                uint8_t inSize = inVal.range(SIZE_DWIDTH - 1, 0);
                outSize += inSize;
                inputWindow.range((inputIdx + factor) * 8 - 1, inputIdx * 8) =
                    inVal.range(SIZE_DWIDTH + DWIDTH - 1, SIZE_DWIDTH);

                ap_uint<DWIDTH> outData = inputWindow.range(DWIDTH - 1, 0);
                inputIdx += inSize;

                // checks per engine end of data or not, use inSize to avoid eos
                blockDone = (inSize == 0);
                is_pending.range(i, i) = (not_first || inSize > 0);

                if (inputIdx >= factor) {
                    outStream << outData;
                    outStreamEos << 0;
                    inputWindow >>= DWIDTH;
                    inputIdx -= factor;
                }
                not_first = true;
            }
        }
    }

    int16_t storedSize = strdSizeStream.read();
    outSize += storedSize;
    for (; storedSize > 0; storedSize -= factor) {
        // adding stored block header
        inputWindow.range((inputIdx + 1) * 8 - 1, inputIdx * 8) = 0;
        inputIdx += 1;
        inputWindow.range((inputIdx + 2) * 8 - 1, inputIdx * 8) = storedSize;
        inputIdx += 2;
        inputWindow.range((inputIdx + 2) * 8 - 1, inputIdx * 8) = ~storedSize;
        inputIdx += 2;
        outSize += 5;

        // process stored block data
        inputWindow.range((inputIdx + factor) * 8 - 1, inputIdx * 8) = strdStream.read();
        ap_uint<DWIDTH> outData = inputWindow.range(DWIDTH - 1, 0);
        inputIdx += factor;
        if (inputIdx >= factor) {
            outStream << outData;
            outStreamEos << 0;
            inputWindow >>= DWIDTH;
            inputIdx -= factor;
        }
    }

    if (outSize <= MIN_BLCK_SIZE) storedSize = strdSizeStream.read();

    if (inputIdx) {
        outStream << inputWindow.range(DWIDTH - 1, 0);
        outStreamEos << 0;
    }

    // send end of stream data
    outStream << 0;
    outStreamEos << 1;
    outSizeStream << outSize;
}

template <int NUM_BLOCKS = 8, int STRATEGY = 0>
void gzipZlibPackerEngine(hls::stream<ap_uint<68> > inStream[NUM_BLOCKS],
                          hls::stream<ap_uint<68> >& packedStream,
                          hls::stream<ap_uint<64> >& strdStream,
                          hls::stream<ap_uint<16> >& strdSizeStream,
                          hls::stream<uint32_t>& fileSizeStream,
                          hls::stream<ap_uint<32> >& checksumStream,
                          hls::stream<ap_uint<4> >& coreIdxStream,
                          hls::stream<bool>& blckEosStream) {
    ap_uint<68> tmpVal = 0;
    ap_uint<4> core = coreIdxStream.read();

    // Header Handling
    if (STRATEGY == 0) { // GZIP
        tmpVal.range(67, 0) = 0x0000000008088B1F8;
        packedStream << tmpVal;
        tmpVal.range(67, 0) = 0x007803004;
        packedStream << tmpVal;
    } else { // ZLIB
        tmpVal.range(67, 0) = 0x01782;
        packedStream << tmpVal;
    }

// Compressed Data Handling
blckHandler:
    for (bool blckEos = blckEosStream.read(); blckEos == false; blckEos = blckEosStream.read()) {
        core %= NUM_BLOCKS;
        bool blockDone = false;
        for (; blockDone == false;) {
#pragma HLS PIPELINE II = 1
            ap_uint<68> inVal = inStream[core].read();
            blockDone = (inVal.range(3, 0) == 0);
            if (blockDone) break;
            packedStream << inVal;
        }
        core++;
    }

    // Stored Block Header
    ap_uint<16> sizeVal = strdSizeStream.read();
    bool strdFlg = (sizeVal != 0);
    if (strdFlg) {
        tmpVal = ~sizeVal;
        tmpVal <<= 16;
        tmpVal.range(15, 0) = sizeVal;
        tmpVal <<= 12;
        tmpVal.range(11, 0) = 5;
        packedStream << tmpVal;
    }

// Stored Block
strdBlck:
    for (uint16_t size = 0; size < sizeVal; size += 8) {
#pragma HLS PIPELINE II = 1
        uint8_t rSize = 8;
        if (rSize + size > sizeVal) rSize = sizeVal - size;
        tmpVal.range(3, 0) = rSize;
        tmpVal.range(67, 4) = strdStream.read();
        packedStream << tmpVal;
    }

    // Checksum Data
    // Last Block Data
    tmpVal.range(67, 0) = 0xffff0000015;
    packedStream << tmpVal;

    tmpVal.range(67, 4) = checksumStream.read();
    tmpVal.range(3, 0) = 4;
    packedStream << tmpVal;
    // Input Size Data
    if (STRATEGY == 0) {
        tmpVal.range(67, 4) = fileSizeStream.read();
        tmpVal.range(3, 0) = 4;
        packedStream << tmpVal;
    }
    tmpVal = 0;
    packedStream << tmpVal;

    for (auto i = 0; i < NUM_BLOCKS; i++) inStream[i].read();
}

template <int DWIDTH = 64, int SIZE_DWIDTH = 4>
void bytePacker(hls::stream<ap_uint<DWIDTH + SIZE_DWIDTH> >& inStream,
                hls::stream<IntVectorStream_dt<8, DWIDTH / 8> >& outStream) {
    constexpr int incr = DWIDTH / 8;
    constexpr uint8_t factor = DWIDTH / 8;
    ap_uint<2 * DWIDTH> inputWindow;

    ap_uint<4> inputIdx = 0;
    bool packerDone = false;
    IntVectorStream_dt<8, DWIDTH / 8> outVal;
    outVal.strobe = incr;

multicorePacker:
    for (; packerDone == false;) {
#pragma HLS PIPELINE II = 1
        assert((inputIdx + factor) <= 2 * (DWIDTH / 8));
        ap_uint<SIZE_DWIDTH + DWIDTH> inVal = inStream.read();

        uint8_t inSize = inVal.range(SIZE_DWIDTH - 1, 0);
        packerDone = (inSize == 0);
        inputWindow.range((inputIdx + factor) * 8 - 1, inputIdx * 8) =
            inVal.range(SIZE_DWIDTH + DWIDTH - 1, SIZE_DWIDTH);

        inputIdx += inSize;

        for (uint16_t k = 0, j = 0; k < DWIDTH; k += incr) {
#pragma HLS UNROLL
            outVal.data[j++] = inputWindow.range(k + 7, k);
        }

        if (inputIdx >= factor) {
            outStream << outVal;
            inputWindow >>= DWIDTH;
            inputIdx -= factor;
        }
    }

    // writing last bytes
    if (inputIdx) {
        for (uint16_t i = 0, j = 0; i < DWIDTH; i += incr) {
#pragma HLS UNROLL
            outVal.data[j++] = inputWindow.range(i + 7, i);
        }
        outVal.strobe = inputIdx;
        outStream << outVal;
    } else {
        // send end of stream data
        outVal.strobe = 0;
        outStream << outVal;
    }
}

template <int NUM_BLOCK = 8, int MAX_FREQ_DWIDTH = 24>
void zlibTreegenScheduler(hls::stream<IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> > lz77InTree[NUM_BLOCK],
                          hls::stream<IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> >& lz77OutTree,
                          hls::stream<uint8_t>& outIdxNum) {
    constexpr int c_litDistCodeCnt = 286 + 30;
    ap_uint<NUM_BLOCK> is_pending;
    bool eos_tmp[NUM_BLOCK];
    for (uint8_t i = 0; i < NUM_BLOCK; i++) {
        is_pending.range(i, i) = 1;
        eos_tmp[i] = false;
    }
    bool read_flag = true;
    IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> inVal;
    while (is_pending) {
        for (uint8_t i = 0; i < NUM_BLOCK; i++) {
            ap_uint<4> j = i % NUM_BLOCK;
            if (eos_tmp[j] == false) {
                inVal = lz77InTree[j].read();
                read_flag = false;
            }
            bool eos = eos_tmp[j] ? eos_tmp[j] : (inVal.strobe == 0);
            is_pending.range(j, j) = eos ? 0 : 1;
            eos_tmp[j] = eos;
            if (!eos) {
                outIdxNum << j;
                for (uint16_t k = 0; k < c_litDistCodeCnt; k++) {
                    if (read_flag) inVal = lz77InTree[j].read();
                    lz77OutTree << inVal;
                    read_flag = true;
                }
            }
        }
    }
    inVal.strobe = 0;
    lz77OutTree << inVal;
    outIdxNum << 0xFF;
}

template <int NUM_BLOCK = 8, int MAX_FREQ_DWIDTH = 24>
void zlibTreegenScheduler(hls::stream<IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> > lz77InTree[NUM_BLOCK],
                          hls::stream<IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> >& lz77OutTree,
                          hls::stream<ap_uint<4> >& coreIdxStream,
                          hls::stream<uint8_t>& outIdxNum) {
    constexpr int c_litDistCodeCnt = 286 + 30;
    ap_uint<NUM_BLOCK> is_pending;
    bool eos_tmp[NUM_BLOCK];
    for (uint8_t i = 0; i < NUM_BLOCK; i++) {
        is_pending.range(i, i) = 1;
        eos_tmp[i] = false;
    }
    bool read_flag = true;
    IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> inVal;
    ap_uint<4> core = coreIdxStream.read();
    while (is_pending) {
        for (uint8_t i = core; i < NUM_BLOCK + core; i++) {
            ap_uint<4> j = i % NUM_BLOCK;
            if (eos_tmp[j] == false) {
                inVal = lz77InTree[j].read();
                read_flag = false;
            }
            bool eos = eos_tmp[j] ? eos_tmp[j] : (inVal.strobe == 0);
            is_pending.range(j, j) = eos ? 0 : 1;
            eos_tmp[j] = eos;
            if (!eos) {
                outIdxNum << j;
                for (uint16_t k = 0; k < c_litDistCodeCnt; k++) {
                    if (read_flag) inVal = lz77InTree[j].read();
                    lz77OutTree << inVal;
                    read_flag = true;
                }
            }
        }
    }
    inVal.strobe = 0;
    lz77OutTree << inVal;
    outIdxNum << 0xFF;
}

template <int NUM_BLOCK = 8>
void zlibTreegenDistributor(hls::stream<DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> > hufCodeStream[NUM_BLOCK],
                            hls::stream<DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> >& hufSerialCodeStream,
                            hls::stream<uint8_t>& inIdxNum) {
    constexpr int c_litDistCodeCnt = 286 + 30;
    DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> hufCodeOut;

tgndst_units_main:
    for (uint8_t i = inIdxNum.read(); i != 0xFF; i = inIdxNum.read()) {
    tgndst_litDist:
        for (uint16_t j = 0; j < c_litDistCodeCnt; j++) {
#pragma HLS PIPELINE II = 1
            hufCodeOut = hufSerialCodeStream.read();
            hufCodeStream[i] << hufCodeOut;
        }
    tgndst_bitlen:
        while (1) {
#pragma HLS PIPELINE II = 1
            hufCodeOut = hufSerialCodeStream.read();
            hufCodeStream[i] << hufCodeOut;
            if (hufCodeOut.data[0].bitlen == 0) break;
        }
    }
    // send eos to all unit
    hufCodeOut.strobe = 0;
tgndst_send_eos:
    for (uint8_t i = 0; i < NUM_BLOCK; ++i) {
        hufCodeStream[i] << hufCodeOut;
    }
}

template <int NUM_BLOCK = 8, int MAX_FREQ_DWIDTH = 24>
void zlibTreegenStreamWrapper(hls::stream<IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> > lz77Tree[NUM_BLOCK],
                              hls::stream<DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> > hufCodeStream[NUM_BLOCK]) {
#pragma HLS dataflow
    hls::stream<IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> > lz77SerialTree("lz77SerialTree");
    hls::stream<DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> > hufSerialCodeStream("hufSerialCodeStream");
    hls::stream<uint8_t> idxNum("idxNum");
#pragma HLS STREAM variable = lz77SerialTree depth = 4
#pragma HLS STREAM variable = hufSerialCodeStream depth = 4
#pragma HLS STREAM variable = idxNum depth = 32

    zlibTreegenScheduler<NUM_BLOCK, MAX_FREQ_DWIDTH>(lz77Tree, lz77SerialTree, idxNum);
    zlibTreegenStream<MAX_FREQ_DWIDTH, 0>(lz77SerialTree, hufSerialCodeStream);
    zlibTreegenDistributor<NUM_BLOCK>(hufCodeStream, hufSerialCodeStream, idxNum);
}

template <int NUM_BLOCK = 8, int MAX_FREQ_DWIDTH = 24>
void zlibTreegenStreamWrapper(hls::stream<IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> > lz77Tree[NUM_BLOCK],
                              hls::stream<DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> > hufCodeStream[NUM_BLOCK],
                              hls::stream<ap_uint<4> >& coreIdxStream) {
#pragma HLS dataflow
    hls::stream<IntVectorStream_dt<MAX_FREQ_DWIDTH, 1> > lz77SerialTree("lz77SerialTree");
    hls::stream<DSVectorStream_dt<HuffmanCode_dt<c_maxBits>, 1> > hufSerialCodeStream("hufSerialCodeStream");
    hls::stream<uint8_t> idxNum("idxNum");
#pragma HLS STREAM variable = lz77SerialTree depth = 4
#pragma HLS STREAM variable = hufSerialCodeStream depth = 4
#pragma HLS STREAM variable = idxNum depth = 16

    zlibTreegenScheduler<NUM_BLOCK, MAX_FREQ_DWIDTH>(lz77Tree, lz77SerialTree, coreIdxStream, idxNum);
    zlibTreegenStream<MAX_FREQ_DWIDTH, 0>(lz77SerialTree, hufSerialCodeStream);
    zlibTreegenDistributor<NUM_BLOCK>(hufCodeStream, hufSerialCodeStream, idxNum);
}

template <int DWIDTH, int SLAVES>
void streamDuplicator(hls::stream<ap_uint<DWIDTH> >& inHlsStream, hls::stream<ap_uint<DWIDTH> > outStream[SLAVES]) {
    ap_uint<DWIDTH> tempVal = inHlsStream.read();
    for (uint8_t i = 0; i < SLAVES; i++) {
#pragma HLS UNROLL
        outStream[i] << tempVal;
    }
}

} // End namespace details
} // End namespace compression
} // End namespace xf

#endif // _XFCOMPRESSION_ZLIB_COMPRESS_DETAILS_HPP_
