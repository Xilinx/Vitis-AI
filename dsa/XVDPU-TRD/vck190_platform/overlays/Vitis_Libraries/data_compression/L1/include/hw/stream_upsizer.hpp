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
#ifndef _XFCOMPRESSION_STREAM_UPSIZER_HPP_
#define _XFCOMPRESSION_STREAM_UPSIZER_HPP_

/**
 * @file stream_upsizer.hpp
 * @brief Header for stream upsizer module.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include "compress_utils.hpp"
#include "hls_stream.h"

#include <ap_int.h>
#include <assert.h>
#include <stdint.h>

namespace xf {
namespace compression {
namespace details {

template <int D_WIDTH>
void sendBuffer(hls::stream<IntVectorStream_dt<D_WIDTH, 1> >& inStream,
                hls::stream<ap_uint<D_WIDTH> >& outStream,
                hls::stream<ap_uint<17> >& outSize) {
    bool last = false;

buffer_top:
    while (!last) {
        last = true;
        ap_uint<17> sizeCntr = 0;
        auto inVal = inStream.read();
        bool loop_continue = (inVal.strobe != 0);
    buffer_main:
        while (loop_continue) {
#pragma HLS PIPELINE II = 1
            last = false;
            loop_continue = (inVal.strobe != 0);
            if (!loop_continue) break;
            outStream << inVal.data[0];
            if (inVal.strobe != 0) {
                inVal = inStream.read();
                sizeCntr++;
            }
        }
        // write out size of up-sized data to terminate the block
        outSize << sizeCntr;
    }
}

template <int IN_WIDTH, int OUT_WIDTH>
void bufferUpsizer(hls::stream<IntVectorStream_dt<IN_WIDTH, 1> >& inStream,
                   hls::stream<ap_uint<OUT_WIDTH> >& outStream,
                   hls::stream<ap_uint<17> >& outSize) {
    constexpr uint8_t c_upsizeFactor = OUT_WIDTH / IN_WIDTH;
    constexpr uint8_t c_inBytes = IN_WIDTH / 8;
    constexpr uint8_t c_outWidth = OUT_WIDTH;
    ap_uint<c_outWidth> outVal;
    bool last = false;

buffer_upsizer_top:
    while (!last) {
        last = true;
        int8_t byteIdx = 0;
        ap_uint<17> sizeCntr = 0;
        auto inVal = inStream.read();
        bool loop_continue = (inVal.strobe != 0);
    buffer_upsizer_main:
        while (loop_continue) {
#pragma HLS PIPELINE II = 1
            last = false;
            if (byteIdx == c_upsizeFactor) {
                // append valid bytes count to output packet
                outStream << outVal;
                byteIdx = 0;
                loop_continue = (inVal.strobe != 0);
            }
            outVal >>= IN_WIDTH;
            outVal.range(c_outWidth - 1, c_outWidth - IN_WIDTH) = inVal.data[0];
            ++byteIdx;
            if (inVal.strobe != 0) {
                inVal = inStream.read();
                sizeCntr += c_inBytes;
            }
        }
        // write out size of up-sized data to terminate the block
        outSize << sizeCntr;
    }
}

template <int IN_WIDTH, int OUT_WIDTH, int BURST_SIZE>
void simpleUpsizer(hls::stream<ap_uint<IN_WIDTH> >& inStream,
                   hls::stream<bool>& inStreamEos,
                   hls::stream<bool>& inFileEos,
                   hls::stream<ap_uint<OUT_WIDTH> >& outStream,
                   hls::stream<bool>& outStreamEos,
                   hls::stream<uint32_t>& outSizeStream) {
    constexpr int c_byteWidth = 8;
    constexpr int c_upsizeFactor = OUT_WIDTH / IN_WIDTH;
    constexpr int c_wordSize = OUT_WIDTH / c_byteWidth;
    constexpr int c_size = BURST_SIZE * c_wordSize;

    while (1) {
        bool eosFile = inFileEos.read();
        if (eosFile == true) break;

        ap_uint<OUT_WIDTH> outBuffer = 0;
        uint32_t byteIdx = 0;
        uint16_t sizeWrite = 0;
        bool eos_flag = false;
    stream_upsizer:
        do {
#pragma HLS PIPELINE II = 1
            if (byteIdx == c_upsizeFactor) {
                outStream << outBuffer;
                outStreamEos << false;
                sizeWrite++;
                if (sizeWrite == BURST_SIZE) {
                    outSizeStream << c_size;
                    sizeWrite = 0;
                }
                byteIdx = 0;
            }
            ap_uint<IN_WIDTH> inValue = inStream.read();
            eos_flag = inStreamEos.read();
            outBuffer.range((byteIdx + 1) * IN_WIDTH - 1, byteIdx * IN_WIDTH) = inValue;
            byteIdx++;
        } while (eos_flag == false);

        if (byteIdx && (eosFile == false)) {
            outStream << outBuffer;
            outStreamEos << true;
            sizeWrite++;
            outSizeStream << (sizeWrite * c_wordSize);
        }
    }
    outSizeStream << 0;
}

template <int IN_WIDTH, int OUT_WIDTH>
void simpleStreamUpsizer(hls::stream<ap_uint<IN_WIDTH> >& inStream,
                         hls::stream<bool>& inStreamEos,
                         hls::stream<uint32_t>& inSizeStream,
                         hls::stream<bool>& inFileEos,
                         hls::stream<ap_uint<OUT_WIDTH> >& outStream,
                         hls::stream<bool>& outStreamEos,
                         hls::stream<ap_uint<4> >& outSizeStream) {
    constexpr int c_byteWidth = 8;
    constexpr int c_upsizeFactor = OUT_WIDTH / IN_WIDTH;
    constexpr int factor = IN_WIDTH / 8;
    uint32_t upsizerCntr = 0;

    while (1) {
        bool eosFile = inFileEos.read();
        if (eosFile == true) break;

        ap_uint<OUT_WIDTH> outBuffer = 0;
        uint8_t byteIdx = 0;
        uint32_t readSize = 0;
        bool eos_flag = false;
    stream_upsizer:
        do {
#pragma HLS PIPELINE II = 1
            if (byteIdx == c_upsizeFactor) {
                readSize += byteIdx * factor;
                outSizeStream << (byteIdx * factor);
                outStream << outBuffer;
                outStreamEos << false;
                byteIdx = 0;
            }
            ap_uint<IN_WIDTH> inValue = inStream.read();
            eos_flag = inStreamEos.read();
            outBuffer.range((byteIdx + 1) * IN_WIDTH - 1, byteIdx * IN_WIDTH) = inValue;
            byteIdx++;
        } while (eos_flag == false);

        uint32_t blockSize = inSizeStream.read();
        uint8_t leftBytes = blockSize - readSize;

        if (byteIdx && (eosFile == false)) {
            outSizeStream << leftBytes;
            outStream << outBuffer;
            outStreamEos << false;
        }
        // send dummy data to indicate end of each block
        outSizeStream << 0;
        outStream << 0;
        outStreamEos << 0;
    }

    outSizeStream << 0;
    outStream << 0;
    outStreamEos << 1;
}

template <int IN_WIDTH, int OUT_WIDTH, int SIZE_DWIDTH = 4>
void simpleStreamUpsizer(hls::stream<IntVectorStream_dt<8, IN_WIDTH / 8> >& inStream,
                         hls::stream<ap_uint<OUT_WIDTH + SIZE_DWIDTH> >& outStream) {
    constexpr uint8_t c_upsizeFactor = OUT_WIDTH / IN_WIDTH;
    constexpr uint8_t c_inBytes = IN_WIDTH / 8;
    ap_uint<IN_WIDTH> inVal;
    ap_uint<OUT_WIDTH> outVal;
    bool last = false;
    ap_uint<4> dsize;

stream_upsizer_top:
    while (!last) {
        last = true;
        uint8_t byteIdx = 0;
        dsize = 0;
        auto inStVal = inStream.read();
        bool loop_continue = (inStVal.strobe != 0);
    stream_upsizer_main:
        while (loop_continue) {
#pragma HLS PIPELINE II = 1
            last = false;
            if (byteIdx == c_upsizeFactor) {
                ap_uint<SIZE_DWIDTH + OUT_WIDTH> tmpVal = outVal;
                tmpVal <<= SIZE_DWIDTH;
                tmpVal.range(SIZE_DWIDTH - 1, 0) = dsize;
                outStream << tmpVal;
                byteIdx = 0;
                dsize = 0;
                loop_continue = (inStVal.strobe != 0);
            }
        upszr_assign_input:
            for (uint8_t b = 0; b < c_inBytes; ++b) {
#pragma HLS UNROLL
#pragma HLS LOOP_TRIPCOUNT min = 0 max = c_inBytes
                if (b < inStVal.strobe) inVal.range(((b + 1) * 8) - 1, b * 8) = inStVal.data[b];
            }
            outVal >>= IN_WIDTH;
            outVal.range(OUT_WIDTH - 1, OUT_WIDTH - IN_WIDTH) = inVal;
            ++byteIdx;
            dsize += inStVal.strobe;
            if (inStVal.strobe != 0) inStVal = inStream.read();
        }
        // end of block/files
        outStream << 0;
    }
}

template <class SIZE_DT, int IN_WIDTH, int OUT_WIDTH>
void streamUpsizer(hls::stream<ap_uint<IN_WIDTH> >& inStream,
                   hls::stream<ap_uint<OUT_WIDTH> >& outStream,
                   SIZE_DT original_size) {
    /**
     * @brief This module reads IN_WIDTH from the input stream and accumulate
     * the consecutive reads until OUT_WIDTH and writes the OUT_WIDTH data to
     * output stream
     *
     * @tparam SIZE_DT stream size class instance
     * @tparam IN_WIDTH input data width
     * @tparam OUT_WIDTH output data width
     *
     * @param inStream input stream
     * @param outStream output stream
     * @param original_size original stream size
     */

    if (original_size == 0) return;

    uint8_t paralle_byte = IN_WIDTH / 8;
    ap_uint<OUT_WIDTH> shift_register;
    uint8_t factor = OUT_WIDTH / IN_WIDTH;
    original_size = (original_size - 1) / paralle_byte + 1;
    uint32_t withAppendedDataSize = (((original_size - 1) / factor) + 1) * factor;

    for (uint32_t i = 0; i < withAppendedDataSize; i++) {
#pragma HLS PIPELINE II = 1
        if (i != 0 && i % factor == 0) {
            outStream << shift_register;
            shift_register = 0;
        }
        if (i < original_size) {
            shift_register.range(OUT_WIDTH - 1, OUT_WIDTH - IN_WIDTH) = inStream.read();
        } else {
            shift_register.range(OUT_WIDTH - 1, OUT_WIDTH - IN_WIDTH) = 0;
        }
        if ((i + 1) % factor != 0) shift_register >>= IN_WIDTH;
    }
    // write last data to stream
    outStream << shift_register;
}

template <int IN_WIDTH, int OUT_WIDTH>
void upsizerEos(hls::stream<ap_uint<IN_WIDTH> >& inStream,
                hls::stream<bool>& inStream_eos,
                hls::stream<ap_uint<OUT_WIDTH> >& outStream,
                hls::stream<bool>& outStream_eos) {
    /**
     * @brief This module reads IN_WIDTH data from input stream based
     * on end of stream and accumulate the consecutive reads until
     * OUT_WIDTH and then writes OUT_WIDTH data to output stream.
     *
     * @tparam IN_WIDTH input data width
     * @tparam OUT_WIDTH output data width
     *
     * @param inStream input stream
     * @param inStream_eos input end of stream flag
     * @param outStream output stream
     * @param outStream_eos output end of stream flag
     */
    // Constants
    const int c_byteWidth = IN_WIDTH;
    const int c_upsizeFactor = OUT_WIDTH / c_byteWidth;
    const int c_inSize = IN_WIDTH / c_byteWidth;

    ap_uint<OUT_WIDTH> outBuffer = 0;
    ap_uint<IN_WIDTH> outBuffer_int[c_upsizeFactor];
#pragma HLS array_partition variable = outBuffer_int dim = 1 complete
    uint32_t byteIdx = 0;
    bool done = false;
    ////printme("%s: reading next data=%d outSize=%d c_inSize=%d\n ",__FUNCTION__, size,outSize,c_inSize);
    outBuffer_int[byteIdx] = inStream.read();
stream_upsizer:
    for (bool eos_flag = inStream_eos.read(); eos_flag == false; eos_flag = inStream_eos.read()) {
#pragma HLS PIPELINE II = 1
        for (int j = 0; j < c_upsizeFactor; j += c_inSize) {
#pragma HLS unroll
            outBuffer.range((j + 1) * c_byteWidth - 1, j * c_byteWidth) = outBuffer_int[j];
        }
        byteIdx += 1;
        ////printme("%s: value=%c, chunk_size = %d and byteIdx=%d\n",__FUNCTION__,(char)tmpValue, chunk_size,byteIdx);
        if (byteIdx >= c_upsizeFactor) {
            outStream << outBuffer;
            outStream_eos << 0;
            byteIdx -= c_upsizeFactor;
        }
        outBuffer_int[byteIdx] = inStream.read();
    }

    if (byteIdx) {
        outStream_eos << 0;
        outStream << outBuffer;
    }
    // end of block

    outStream << 0;
    outStream_eos << 1;
    // printme("%s:Ended \n",__FUNCTION__);
}

template <class SIZE_DT, int IN_WIDTH, int OUT_WIDTH>
void upsizer_sizestream(hls::stream<ap_uint<IN_WIDTH> >& inStream,
                        hls::stream<SIZE_DT>& inStreamSize,
                        hls::stream<ap_uint<OUT_WIDTH> >& outStream,
                        hls::stream<SIZE_DT>& outStreamSize) {
    // Constants
    const int c_byte_width = 8; // 8bit is each BYTE
    const int c_upsize_factor = OUT_WIDTH / c_byte_width;
    const int c_in_size = IN_WIDTH / c_byte_width;

    ap_uint<2 * OUT_WIDTH> outBuffer = 0; // Declaring double buffers
    uint32_t byteIdx = 0;
    // printme("%s: factor=%d\n",__FUNCTION__,c_upsize_factor);
    for (SIZE_DT size = inStreamSize.read(); size != 0; size = inStreamSize.read()) {
        // rounding off the output size
        uint16_t outSize = ((size + byteIdx) / c_upsize_factor) * c_upsize_factor;
        if (outSize) {
            outStreamSize << outSize;
        }
    ////printme("%s: reading next data=%d outSize=%d c_in_size=%d\n ",__FUNCTION__, size,outSize,c_in_size);
    stream_upsizer:
        for (int i = 0; i < size; i += c_in_size) {
#pragma HLS PIPELINE II = 1
            int chunk_size = c_in_size;
            if (chunk_size + i > size) chunk_size = size - i;
            ap_uint<IN_WIDTH> tmpValue = inStream.read();
            outBuffer.range((byteIdx + c_in_size) * c_byte_width - 1, byteIdx * c_byte_width) = tmpValue;
            byteIdx += chunk_size;
            ////printme("%s: value=%c, chunk_size = %d and byteIdx=%d\n",__FUNCTION__,(char)tmpValue,
            /// chunk_size,byteIdx);
            if (byteIdx >= c_upsize_factor) {
                outStream << outBuffer.range(OUT_WIDTH - 1, 0);
                outBuffer >>= OUT_WIDTH;
                byteIdx -= c_upsize_factor;
            }
        }
    }
    if (byteIdx) {
        outStreamSize << byteIdx;
        ////printme("sent outSize %d \n", byteIdx);
        outStream << outBuffer.range(OUT_WIDTH - 1, 0);
    }
    // end of block
    outStreamSize << 0;
    // printme("%s:Ended \n",__FUNCTION__);
}

template <int OUT_WIDTH, int PACK_WIDTH>
void streamUpsizerP2P(hls::stream<ap_uint<PACK_WIDTH> >& inStream,
                      hls::stream<ap_uint<OUT_WIDTH> >& outStream,
                      hls::stream<uint32_t>& inStreamSize,
                      hls::stream<uint32_t>& outStreamSize) {
    const int c_byte_width = 8;
    const int c_upsize_factor = OUT_WIDTH / c_byte_width;
    const int c_in_size = PACK_WIDTH / c_byte_width;

    // Declaring double buffers
    ap_uint<2 * OUT_WIDTH> outBuffer = 0;
    uint32_t byteIdx = 0;

    for (int size = inStreamSize.read(); size != 0; size = inStreamSize.read()) {
        // printf("Size %d \n", size);
        // Rounding off the output size
        uint32_t outSize = (size * c_byte_width + byteIdx) / PACK_WIDTH;

        if (outSize) outStreamSize << outSize;
    streamUpsizer:
        for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II = 1
            // printf("val/size %d/%d \n", i, size);
            ap_uint<PACK_WIDTH> tmpValue = inStream.read();
            outBuffer.range((byteIdx + c_in_size) * c_byte_width - 1, byteIdx * c_byte_width) = tmpValue;
            byteIdx += c_byte_width;

            if (byteIdx >= c_upsize_factor) {
                outStream << outBuffer.range(OUT_WIDTH - 1, 0);
                outBuffer >>= OUT_WIDTH;
                byteIdx -= c_upsize_factor;
            }
        }
    }

    if (byteIdx) {
        outStreamSize << 1;
        outStream << outBuffer.range(OUT_WIDTH - 1, 0);
    }
    // printf("%s Done \n", __FUNCTION__);
    // end of block
    outStreamSize << 0;
}

} // namespace details
} // namespace compression
} // namespace xf

#endif // _XFCOMPRESSION_STREAM_UPSIZER_HPP_
