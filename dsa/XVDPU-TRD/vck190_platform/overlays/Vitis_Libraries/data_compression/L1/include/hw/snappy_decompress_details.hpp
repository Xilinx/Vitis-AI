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
#ifndef _XFCOMPRESSION_SNAPPY_DECOMPRESS_DETAILS_HPP_
#define _XFCOMPRESSION_SNAPPY_DECOMPRESS_DETAILS_HPP_

/**
 * @file snappy_decompress_details.hpp
 * @brief Internal modules used for snappy decompression
 *
 * This file is part of Vitis Data Compression Library.
 */
#include "hls_stream.h"

#include <ap_int.h>
#include <assert.h>
#include <stdint.h>
#include "snappy_decompress.hpp"
#include "lz_decompress.hpp"

namespace xf {
namespace compression {

namespace details {

template <int PARALLEL_BYTES, class SIZE_DT = ap_uint<17> >
static void snappyHeaderProcessing(hls::stream<ap_uint<PARALLEL_BYTES * 8> >& inStream,
                                   hls::stream<ap_uint<PARALLEL_BYTES * 8> >& outStream,
                                   hls::stream<dt_blockInfo>& blockInfoStream,
                                   const uint32_t inputSize) {
    // Snappy Header states
    enum snappyDecompressHeaderStates { READ_COMP_LEN, WRITE_DATA };
    enum snappyDecompressHeaderStates nextState = READ_COMP_LEN;

    const int c_parallelBit = PARALLEL_BYTES * 8;
    ap_uint<8 * c_parallelBit> inputWindow;
    ap_uint<c_parallelBit> outStreamValue = 0;

    uint32_t readBytes = 0, processedBytes = 0;
    SIZE_DT origCompLen = 0, compLen = 0;
    uint8_t inputIdx = 0;
    bool outFlag = false;

    // process first 10 bytes
    ap_uint<c_parallelBit> temp1 = inStream.read();
    readBytes += PARALLEL_BYTES;
    processedBytes += PARALLEL_BYTES;

    if (PARALLEL_BYTES < 8) {
        temp1 = inStream.read();
        readBytes += PARALLEL_BYTES;
        processedBytes += PARALLEL_BYTES;
    }

    for (uint8_t i = 0; i < 8; i++) {
#pragma HLS PIPELINE II = 1
        inputWindow.range((i + 1) * c_parallelBit - 1, i * c_parallelBit) = inStream.read();
        readBytes += PARALLEL_BYTES;
    }

    inputIdx += 2;
    dt_blockInfo blockInfo;
    uint8_t incrInputIdx = 0;

    while ((processedBytes + inputIdx) < inputSize) {
        if (inputIdx >= PARALLEL_BYTES) {
            inputWindow >>= c_parallelBit;
            inputIdx = inputIdx - PARALLEL_BYTES;
            processedBytes += PARALLEL_BYTES;
            if (readBytes < inputSize) {
                ap_uint<c_parallelBit> input = inStream.read();
                inputWindow.range(8 * c_parallelBit - 1, 7 * c_parallelBit) = input;
                readBytes += PARALLEL_BYTES;
            }
        }

        SIZE_DT chunkSize = 0;

        ap_uint<32> inValue = inputWindow >> (inputIdx * 8);
        chunkSize = inValue.range(31, 8);

        origCompLen = chunkSize - 4;
        compLen = origCompLen;

        // 4 bytes processed and remaining 4 bytes skipped
        incrInputIdx = 8;

        uint8_t chunkIdx = inValue.range(7, 0);

        if (chunkIdx == 0x01) {
            blockInfo.storedBlock = 1;
        } else {
            blockInfo.storedBlock = 0;
            uint8_t blkSizeProcessedBytes = 0;
            ap_uint<16> value = inputWindow >> ((inputIdx + incrInputIdx) * 8);

            bool c0 = ((value.range(7, 7)) == 1);
            bool c1 = ((value.range(15, 15)) == 1);
            if (c0 & c1) {
                incrInputIdx = 11;
                blkSizeProcessedBytes = 3;
            } else if (c0) {
                incrInputIdx = 10;
                blkSizeProcessedBytes = 2;
            } else {
                incrInputIdx = 9;
                blkSizeProcessedBytes = 1;
            }
            origCompLen = origCompLen - blkSizeProcessedBytes;
            compLen = origCompLen;
        }

        inputIdx += incrInputIdx;
        blockInfo.compressSize = origCompLen;
        // write blockInfo to stream
        blockInfoStream << blockInfo;

        SIZE_DT len = compLen;
        for (uint32_t blockData = 0; blockData < compLen; blockData += PARALLEL_BYTES) {
#pragma HLS PIPELINE II = 1
            if (inputIdx >= PARALLEL_BYTES) {
                inputWindow >>= c_parallelBit;
                inputIdx = inputIdx - PARALLEL_BYTES;
                processedBytes += PARALLEL_BYTES;
                if (readBytes < inputSize) {
                    ap_uint<c_parallelBit> input = inStream.read();
                    inputWindow.range(8 * c_parallelBit - 1, 7 * c_parallelBit) = input;
                    readBytes += PARALLEL_BYTES;
                }
            }

            outStreamValue = inputWindow >> (inputIdx * 8);
            outStream << outStreamValue;

            if (len >= PARALLEL_BYTES) {
                inputIdx += PARALLEL_BYTES;
                len -= PARALLEL_BYTES;
            } else {
                inputIdx += len;
                len = 0;
            }
        }
    }
    blockInfo.compressSize = 0;
    // writing 0 to indicate end of data
    blockInfoStream << blockInfo;
}

template <int NUM_BLOCKS, int PARALLEL_BYTES, class SIZE_DT = ap_uint<17> >
static void snappyMultiBlockHeaderProcessing(hls::stream<ap_uint<PARALLEL_BYTES * 8> >& inStream,
                                             hls::stream<ap_uint<PARALLEL_BYTES * 8> > outStream[NUM_BLOCKS],
                                             hls::stream<dt_blockInfo> blockInfoStream[NUM_BLOCKS],
                                             hls::stream<uint32_t>& inSizeStream) {
    for (uint32_t inputSize = inSizeStream.read(); inputSize != 0; inputSize = inSizeStream.read()) {
        // Snappy Header states
        enum snappyDecompressHeaderStates { READ_COMP_LEN, WRITE_DATA };
        enum snappyDecompressHeaderStates nextState = READ_COMP_LEN;

        uint32_t numBlocks = 0;
        const int c_parallelBit = PARALLEL_BYTES * 8;
        ap_uint<8 * c_parallelBit> inputWindow;
        ap_uint<c_parallelBit> outStreamValue = 0;

        uint32_t readBytes = 0, processedBytes = 0;
        SIZE_DT origCompLen = 0, compLen = 0;
        uint8_t inputIdx = 0;
        bool outFlag = false;

        // process first 10 bytes
        ap_uint<c_parallelBit> temp1 = inStream.read();
        readBytes += PARALLEL_BYTES;
        processedBytes += PARALLEL_BYTES;

        if (PARALLEL_BYTES < 8) {
            temp1 = inStream.read();
            readBytes += PARALLEL_BYTES;
            processedBytes += PARALLEL_BYTES;
        }

        for (uint8_t i = 0; i < 8; i++) {
#pragma HLS PIPELINE II = 1
            inputWindow.range((i + 1) * c_parallelBit - 1, i * c_parallelBit) = inStream.read();
            readBytes += PARALLEL_BYTES;
        }

        inputIdx += 2;
        uint8_t incrInputIdx = 0;

        while ((processedBytes + inputIdx) < inputSize) {
            dt_blockInfo blockInfo;
            if (inputIdx >= PARALLEL_BYTES) {
                inputWindow >>= c_parallelBit;
                inputIdx = inputIdx - PARALLEL_BYTES;
                processedBytes += PARALLEL_BYTES;
                if (readBytes < inputSize) {
                    ap_uint<c_parallelBit> input = inStream.read();
                    inputWindow.range(8 * c_parallelBit - 1, 7 * c_parallelBit) = input;
                    readBytes += PARALLEL_BYTES;
                }
            }

            SIZE_DT chunkSize = 0;

            ap_uint<32> inValue = inputWindow >> (inputIdx * 8);
            chunkSize = inValue.range(31, 8);

            origCompLen = chunkSize - 4;
            compLen = origCompLen;

            // 4 bytes processed and remaining 4 bytes skipped
            incrInputIdx = 8;

            uint8_t chunkIdx = inValue.range(7, 0);

            if (chunkIdx == 0x01) {
                blockInfo.storedBlock = 1;
            } else {
                blockInfo.storedBlock = 0;
                uint8_t blkSizeProcessedBytes = 0;
                ap_uint<16> value = inputWindow >> ((inputIdx + incrInputIdx) * 8);

                bool c0 = ((value.range(7, 7)) == 1);
                bool c1 = ((value.range(15, 15)) == 1);
                if (c0 & c1) {
                    incrInputIdx = 11;
                    blkSizeProcessedBytes = 3;
                } else if (c0) {
                    incrInputIdx = 10;
                    blkSizeProcessedBytes = 2;
                } else {
                    incrInputIdx = 9;
                    blkSizeProcessedBytes = 1;
                }
                origCompLen = origCompLen - blkSizeProcessedBytes;
                compLen = origCompLen;
            }

            inputIdx += incrInputIdx;
            uint8_t core = numBlocks % NUM_BLOCKS;
            blockInfo.compressSize = origCompLen;
            // write blockInfo to stream
            blockInfoStream[core] << blockInfo;

            SIZE_DT len = compLen;
            for (uint32_t blockData = 0; blockData < compLen; blockData += PARALLEL_BYTES) {
#pragma HLS PIPELINE II = 1
                if (inputIdx >= PARALLEL_BYTES) {
                    inputWindow >>= c_parallelBit;
                    inputIdx = inputIdx - PARALLEL_BYTES;
                    processedBytes += PARALLEL_BYTES;
                    if (readBytes < inputSize) {
                        ap_uint<c_parallelBit> input = inStream.read();
                        inputWindow.range(8 * c_parallelBit - 1, 7 * c_parallelBit) = input;
                        readBytes += PARALLEL_BYTES;
                    }
                }

                outStreamValue = inputWindow >> (inputIdx * 8);
                outStream[core] << outStreamValue;

                if (len >= PARALLEL_BYTES) {
                    inputIdx += PARALLEL_BYTES;
                    len -= PARALLEL_BYTES;
                } else {
                    inputIdx += len;
                    len = 0;
                }
            }
            numBlocks++;
        }
        dt_blockInfo blockInfo;
        blockInfo.compressSize = 0;

        // writing 0 to indicate end of data
        for (int core = 0; core < NUM_BLOCKS; core++) {
#pragma HLS UNROLL
            blockInfoStream[core] << blockInfo;
        }
    }
}

// Gather data from each num block and pack it correctly
// to produce the data in order
template <int NUM_BLOCKS, int PARALLEL_BYTES, int BLOCK_SIZE = 64, class SIZE_DT = ap_uint<17> >
void lzMultiBlockPacker(hls::stream<ap_uint<(PARALLEL_BYTES * 8) + 8> > lzDataStream[NUM_BLOCKS],
                        hls::stream<uint32_t> lzSizeStream[NUM_BLOCKS],
                        hls::stream<ap_uint<(PARALLEL_BYTES * 8) + 8> >& outStream,
                        hls::stream<uint32_t>& outSizeStream) {
    uint32_t outSize = 0;
    const uint8_t c_parallelBit = PARALLEL_BYTES * 8;
    const uint8_t c_streamWidth = (PARALLEL_BYTES * 8) + 8;
    uint32_t blockSizeInBytes = BLOCK_SIZE * 1024;
    ap_uint<NUM_BLOCKS> is_pending;
    bool done = false;

    for (uint8_t i = 0; i < NUM_BLOCKS; i++) {
#pragma HLS UNROLL
        is_pending.range(i, i) = 1;
    }

    while (is_pending) {
        for (int i = 0; i < NUM_BLOCKS; i++) {
            for (SIZE_DT read = 0; ((read < blockSizeInBytes) && (is_pending.range(i, i) == true));
                 read += PARALLEL_BYTES) {
#pragma HLS PIPELINE II = 1
                ap_uint<c_streamWidth> outData = lzDataStream[i].read();
                bool eosFlag = outData.range(c_streamWidth - 1, c_parallelBit);
                if (eosFlag == false) {
                    outStream << outData;
                } else {
                    is_pending.range(i, i) = 0;
                }
            }
        }
    }

    ap_uint<c_streamWidth> outData = 0;
    outData.range(c_streamWidth - 1, c_parallelBit) = 1;
    outStream << outData;

    for (int i = 0; i < NUM_BLOCKS; i++) {
#pragma HLS PIPELINE II = 1
        outSize += lzSizeStream[i].read();
    }
    outSizeStream << outSize;
}

} // namespace details

// Process block by block and generate uncompressed data
template <int NUM_BLOCKS, int PARALLEL_BYTES, int HISTORY_SIZE, int BLOCK_SIZE = 64, class SIZE_DT = ap_uint<17> >
void snappyBlockDecoder(hls::stream<ap_uint<PARALLEL_BYTES * 8> >& blockDataStream,
                        hls::stream<dt_blockInfo>& blockInfoStream,
                        hls::stream<ap_uint<(PARALLEL_BYTES * 8) + 8> >& lzDataStream,
                        hls::stream<uint32_t>& lzSizeStream) {
    typedef ap_uint<16> offset_dt;
    hls::stream<SIZE_DT> litlenStream("litlenStream");
    hls::stream<ap_uint<PARALLEL_BYTES * 8> > litStream("litStream");
    hls::stream<offset_dt> offsetStream("offsetStream");
    hls::stream<SIZE_DT> matchLenStream("matchLenStream");

#pragma HLS STREAM variable = litlenStream depth = 32
#pragma HLS STREAM variable = litStream depth = 32
#pragma HLS STREAM variable = offsetStream depth = 32
#pragma HLS STREAM variable = matchLenStream depth = 32

#pragma HLS BIND_STORAGE variable = litlenStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = litStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = offsetStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = matchLenStream type = FIFO impl = SRL

#pragma HLS dataflow
    // last template arg as true to run block decom
    snappyMultiByteDecompress<PARALLEL_BYTES, SIZE_DT>(blockDataStream, litlenStream, litStream, offsetStream,
                                                       matchLenStream, blockInfoStream);
    lzMultiByteDecoder<PARALLEL_BYTES, HISTORY_SIZE, SIZE_DT>(litlenStream, litStream, offsetStream, matchLenStream,
                                                              lzDataStream, lzSizeStream);
}

template <int NUM_BLOCKS, int PARALLEL_BYTES, int HISTORY_SIZE, int BLOCK_SIZE = 64, class SIZE_DT = ap_uint<17> >
void snappyMultiCoreDecompress(hls::stream<ap_uint<PARALLEL_BYTES * 8> >& inStream,
                               hls::stream<uint32_t>& inSizeStream,
                               hls::stream<ap_uint<(PARALLEL_BYTES * 8) + 8> >& outStream,
                               hls::stream<uint32_t>& outSizeStream) {
    const uint8_t c_streamWidth = (PARALLEL_BYTES * 8) + 8;
    typedef ap_uint<c_streamWidth> uintV_t;
    typedef ap_uint<PARALLEL_BYTES * 8> uintS_t;
    constexpr int depthBlockSizeInBytes = (BLOCK_SIZE * 1024) / PARALLEL_BYTES;

    hls::stream<uintS_t> blockDataStream[NUM_BLOCKS];
    hls::stream<dt_blockInfo> blockInfoStream[NUM_BLOCKS];
    hls::stream<uintV_t> lzDataStream[NUM_BLOCKS];
    hls::stream<uint32_t> lzSizeStream[NUM_BLOCKS];
#pragma HLS STREAM variable = blockDataStream depth = depthBlockSizeInBytes
#pragma HLS STREAM variable = blockInfoStream depth = 32
#pragma HLS STREAM variable = lzDataStream depth = depthBlockSizeInBytes
#pragma HLS STREAM variable = lzSizeStream depth = 32

#pragma HLS BIND_STORAGE variable = blockDataStream type = FIFO impl = URAM
#pragma HLS BIND_STORAGE variable = blockInfoStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = lzDataStream type = FIFO impl = URAM
#pragma HLS BIND_STORAGE variable = lzSizeStream type = FIFO impl = SRL

#pragma HLS dataflow
    details::snappyMultiBlockHeaderProcessing<NUM_BLOCKS, PARALLEL_BYTES, SIZE_DT>(inStream, blockDataStream,
                                                                                   blockInfoStream, inSizeStream);
    for (int i = 0; i < NUM_BLOCKS; i++) {
#pragma HLS UNROLL
        snappyBlockDecoder<NUM_BLOCKS, PARALLEL_BYTES, HISTORY_SIZE, BLOCK_SIZE, SIZE_DT>(
            blockDataStream[i], blockInfoStream[i], lzDataStream[i], lzSizeStream[i]);
    }
    details::lzMultiBlockPacker<NUM_BLOCKS, PARALLEL_BYTES, BLOCK_SIZE, SIZE_DT>(lzDataStream, lzSizeStream, outStream,
                                                                                 outSizeStream);
}

template <int PARALLEL_BYTES, int HISTORY_SIZE, class SIZE_DT = ap_uint<17> >
void snappyDecompressEngine(hls::stream<ap_uint<PARALLEL_BYTES * 8> >& inStream,
                            hls::stream<ap_uint<(PARALLEL_BYTES * 8) + 8> >& outStream,
                            hls::stream<uint32_t>& outSizeStream,
                            const uint32_t _input_size) {
    typedef ap_uint<PARALLEL_BYTES * 8> uintV_t;
    typedef ap_uint<16> offset_dt;
    uint32_t input_size1 = _input_size;
    hls::stream<uintV_t> headerStream("headerStream");
    hls::stream<SIZE_DT> litlenStream("litlenStream");
    hls::stream<uintV_t> litStream("litStream");
    hls::stream<offset_dt> offsetStream("offsetStream");
    hls::stream<SIZE_DT> matchLenStream("matchLenStream");
    hls::stream<dt_blockInfo> blockInfoStream("blockInfoStream");
#pragma HLS STREAM variable = headerStream depth = 32
#pragma HLS STREAM variable = blockInfoStream depth = 32
#pragma HLS STREAM variable = litlenStream depth = 32
#pragma HLS STREAM variable = litStream depth = 32
#pragma HLS STREAM variable = offsetStream depth = 32
#pragma HLS STREAM variable = matchLenStream depth = 32

#pragma HLS BIND_STORAGE variable = headerStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = blockInfoStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = litlenStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = litStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = offsetStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = matchLenStream type = FIFO impl = SRL

#pragma HLS dataflow
    details::snappyHeaderProcessing<PARALLEL_BYTES, SIZE_DT>(inStream, headerStream, blockInfoStream, input_size1);
    snappyMultiByteDecompress<PARALLEL_BYTES, SIZE_DT>(headerStream, litlenStream, litStream, offsetStream,
                                                       matchLenStream, blockInfoStream);
    lzMultiByteDecoder<PARALLEL_BYTES, HISTORY_SIZE, SIZE_DT>(litlenStream, litStream, offsetStream, matchLenStream,
                                                              outStream, outSizeStream);
}

template <int PARALLEL_BYTES, int HISTORY_SIZE, class SIZE_DT = ap_uint<17> >
void snappyDecompressCoreEngine(hls::stream<ap_uint<PARALLEL_BYTES * 8> >& inStream,
                                hls::stream<ap_uint<(PARALLEL_BYTES * 8) + 8> >& outStream,
                                hls::stream<uint32_t>& outSizeStream,
                                hls::stream<uint32_t>& blockSizeStream) {
    typedef ap_uint<PARALLEL_BYTES * 8> uintV_t;
    typedef ap_uint<16> offset_dt;
    hls::stream<SIZE_DT> litlenStream("litlenStream");
    hls::stream<SIZE_DT> matchLenStream("matchLenStream");
    hls::stream<offset_dt> offsetStream("offsetStream");
    hls::stream<uintV_t> litStream("litStream");
    hls::stream<dt_blockInfo> blockInfoStream("blockInfoStream");
#pragma HLS STREAM variable = litlenStream depth = 32
#pragma HLS STREAM variable = offsetStream depth = 32
#pragma HLS STREAM variable = matchLenStream depth = 32
#pragma HLS STREAM variable = litStream depth = 32
#pragma HLS STREAM variable = blockInfoStream depth = 32

#pragma HLS BIND_STORAGE variable = litlenStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = offsetStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = matchLenStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = litStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = blockInfoStream type = FIFO impl = SRL

    dt_blockInfo blockInfo;
    for (int i = 0; i < 2; i++) {
        blockInfo.compressSize = blockSizeStream.read();
        blockInfo.storedBlock = 0;

        blockInfoStream << blockInfo;
    }
#pragma HLS dataflow
    // last template arg true to run block decom
    snappyMultiByteDecompress<PARALLEL_BYTES, SIZE_DT, true>(inStream, litlenStream, litStream, offsetStream,
                                                             matchLenStream, blockInfoStream);
    lzMultiByteDecoder<PARALLEL_BYTES, HISTORY_SIZE, SIZE_DT>(litlenStream, litStream, offsetStream, matchLenStream,
                                                              outStream, outSizeStream);
}

} // namespace compression
} // namespace xf

#endif // _XFCOMPRESSION_SNAPPY_DECOMPRESS_DETAILS_HPP_
