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
#ifndef _XFCOMPRESSION_STREAM_DOWNSIZER_HPP_
#define _XFCOMPRESSION_STREAM_DOWNSIZER_HPP_

/**
 * @file stream_downsizer.hpp
 * @brief Header for stream downsizer module.
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

template <int IN_DATAWIDTH>
void receiveBuffer(hls::stream<ap_uint<IN_DATAWIDTH> >& inStream,
                   hls::stream<IntVectorStream_dt<IN_DATAWIDTH, 1> >& outStream,
                   hls::stream<ap_uint<17> >& inputSize) {
    IntVectorStream_dt<IN_DATAWIDTH, 1> outVal;

buffer_top:
    while (1) {
        ap_uint<17> inSize = inputSize.read();
        ap_uint<IN_DATAWIDTH> inVal = 0;
        // proceed further if valid size
        if (inSize == 0) break;
        auto outSizeV = inSize;
        outVal.strobe = 1;
    buffer_assign:
        for (auto i = 0; i < outSizeV; i++) {
#pragma HLS PIPELINE II = 1
            inVal = inStream.read();
            outVal.data[0] = inVal;
            outStream << outVal;
        }
        // Block end Condition
        outVal.strobe = 0;
        outStream << outVal;
    }
    // File end Condition
    outVal.strobe = 0;
    outStream << outVal;
}

template <int IN_DATAWIDTH, int OUT_DATAWIDTH>
void bufferDownsizer(hls::stream<ap_uint<IN_DATAWIDTH> >& inStream,
                     hls::stream<IntVectorStream_dt<OUT_DATAWIDTH, 1> >& outStream,
                     hls::stream<ap_uint<17> >& inputSize) {
    constexpr int16_t c_factor = IN_DATAWIDTH / OUT_DATAWIDTH;
    constexpr int16_t c_outWord = OUT_DATAWIDTH / 8;
    IntVectorStream_dt<OUT_DATAWIDTH, 1> outVal;

downsizer_top:
    while (1) {
        ap_uint<17> inSize = inputSize.read();
        ap_uint<IN_DATAWIDTH> inVal = 0;
        ap_uint<17> cntr = 0;
        // proceed further if valid size
        if (inSize == 0) break;
        auto outSizeV = ((inSize - 1) / c_outWord) + 1;
        outVal.strobe = 1;
    downsizer_assign:
        for (auto i = 0; i < outSizeV; i += c_outWord) {
#pragma HLS PIPELINE II = 1
            if (cntr % c_factor == 0) inVal = inStream.read();
            outVal.data[0] = inVal.range(OUT_DATAWIDTH - 1, 0);
            inVal >>= OUT_DATAWIDTH;
            outStream << outVal;
            cntr++;
        }
        // Block end Condition
        outVal.strobe = 0;
        outStream << outVal;
    }
    // File end Condition
    outVal.strobe = 0;
    outStream << outVal;
}

template <int IN_DATAWIDTH, int OUT_DATAWIDTH, int SIZE_DWIDTH = 4>
void bufferDownsizer(hls::stream<ap_uint<IN_DATAWIDTH + SIZE_DWIDTH> >& inStream,
                     hls::stream<IntVectorStream_dt<OUT_DATAWIDTH, 1> >& outStream) {
    constexpr int16_t c_factor = IN_DATAWIDTH / OUT_DATAWIDTH;
    constexpr int16_t c_outWord = OUT_DATAWIDTH / 8;
    IntVectorStream_dt<OUT_DATAWIDTH, 1> outVal;

downsizer_top:
    while (1) {
        ap_uint<SIZE_DWIDTH> dsize = 0;
        auto inVal = inStream.read();
        // proceed further if valid size
        ap_uint<SIZE_DWIDTH> inSize = inVal.range(SIZE_DWIDTH - 1, 0);
        if (inSize == 0) break;
        auto outSizeV = ((inSize - 1) / c_outWord) + 1;
        outVal.strobe = 1;
    downsizer_assign:
        while (inSize > 0) {
#pragma HLS PIPELINE II = 1
            outVal.data[0] = inVal.range(OUT_DATAWIDTH + SIZE_DWIDTH - 1, SIZE_DWIDTH);
            inVal >>= OUT_DATAWIDTH;
            outStream << outVal;
            dsize += c_outWord;
            if (dsize == outSizeV) {
                inVal = inStream.read();
                inSize = inVal.range(SIZE_DWIDTH - 1, 0);
                dsize = 0;
                outSizeV = ((inSize - 1) / c_outWord) + 1;
            }
        }
        // Block end Condition
        outVal.strobe = 0;
        outStream << outVal;
    }
    // File end Condition
    outVal.strobe = 0;
    outStream << outVal;
}

template <int IN_DATAWIDTH, int OUT_DATAWIDTH, int SIZE_DWIDTH = 4>
void bufferDownsizerVec(hls::stream<ap_uint<IN_DATAWIDTH + SIZE_DWIDTH> >& inStream,
                        hls::stream<IntVectorStream_dt<8, OUT_DATAWIDTH / 8> >& outStream) {
    constexpr uint16_t c_factor = IN_DATAWIDTH / OUT_DATAWIDTH;
    constexpr uint8_t c_outWord = OUT_DATAWIDTH / 8;
    constexpr uint8_t c_outDataHigh = OUT_DATAWIDTH + SIZE_DWIDTH - 1;
    IntVectorStream_dt<8, c_outWord> outVal;

downsizer_top:
    while (1) {
        auto inVal = inStream.read();
        // proceed further if valid size
        ap_uint<SIZE_DWIDTH> inSize = inVal.range(SIZE_DWIDTH - 1, 0);
        if (inSize == 0) break;
    downsizer_assign:
        while (inSize > 0) {
#pragma HLS PIPELINE II = 1
            ap_uint<OUT_DATAWIDTH> outReg = inVal.range(c_outDataHigh, SIZE_DWIDTH);
            inVal >>= OUT_DATAWIDTH;
            outVal.strobe = ((inSize < c_outWord) ? (uint8_t)inSize : c_outWord);
            for (uint8_t i = 0; i < c_outWord; ++i) {
#pragma HLS UNROLL
                outVal.data[i] = outReg.range((i * 8) + 7, i * 8);
            }
            outStream << outVal;
            inSize -= outVal.strobe;
            if (inSize == 0) {
                inVal = inStream.read();
                inSize = inVal.range(SIZE_DWIDTH - 1, 0);
            }
        }
        // Block end Condition
        outVal.strobe = 0;
        outStream << outVal;
    }
    // File end Condition
    outVal.strobe = 0;
    outStream << outVal;
}

template <int IN_DATAWIDTH, int OUT_DATAWIDTH>
void simpleStreamDownSizer(hls::stream<ap_uint<IN_DATAWIDTH> >& inStream,
                           hls::stream<uint16_t>& inSizeStream,
                           hls::stream<ap_uint<OUT_DATAWIDTH> >& outStream) {
    const int c_byteWidth = 8;
    const int c_inputWord = IN_DATAWIDTH / c_byteWidth;
    const int c_outWord = OUT_DATAWIDTH / c_byteWidth;
    const int factor = c_inputWord / c_outWord;
    ap_uint<IN_DATAWIDTH> inBuffer = 0;

downsizer_top:
    for (uint16_t inSize = inSizeStream.read(); inSize != 0; inSize = inSizeStream.read()) {
        uint16_t outSizeV = (inSize - 1) / c_outWord + 1;
    downsizer_assign:
        for (uint16_t itr = 0; itr < outSizeV; itr++) {
#pragma HLS PIPELINE II = 1
            int idx = itr % factor;
            if (idx == 0) inBuffer = inStream.read();
            ap_uint<OUT_DATAWIDTH> tmpValue = inBuffer.range((idx + 1) * OUT_DATAWIDTH - 1, idx * OUT_DATAWIDTH);
            outStream << tmpValue;
        }
    }
}

template <int IN_DATAWIDTH, int OUT_DATAWIDTH, class SIZE_DT = uint32_t>
void streamDownSizerSize(hls::stream<ap_uint<IN_DATAWIDTH> >& inStream,
                         hls::stream<SIZE_DT>& inSizeStream,
                         hls::stream<ap_uint<OUT_DATAWIDTH> >& outStream,
                         hls::stream<uint32_t>& outSizeStream) {
    constexpr int c_byteWidth = 8;
    constexpr int c_inputWord = IN_DATAWIDTH / c_byteWidth;
    constexpr int c_outWord = OUT_DATAWIDTH / c_byteWidth;
    constexpr int factor = c_inputWord / c_outWord;
    ap_uint<IN_DATAWIDTH> inBuffer = 0;

downsizer_top:
    for (SIZE_DT inSize = inSizeStream.read(); inSize != 0; inSize = inSizeStream.read()) {
        outSizeStream << inSize;
        SIZE_DT outSizeV = (inSize - 1) / c_outWord + 1;
    downsizer_assign:
        for (SIZE_DT itr = 0; itr < outSizeV; itr++) {
#pragma HLS PIPELINE II = 1
            SIZE_DT idx = itr % factor;
            if (idx == 0) inBuffer = inStream.read();
            ap_uint<OUT_DATAWIDTH> tmpValue = inBuffer.range((idx + 1) * OUT_DATAWIDTH - 1, idx * OUT_DATAWIDTH);
            outStream << tmpValue;
        }
    }
    outSizeStream << 0;
}

template <int IN_DATAWIDTH, int OUT_DATAWIDTH, int SIZE_DWIDTH = 24>
void streamDownSizerSize(hls::stream<ap_uint<IN_DATAWIDTH> >& inStream,
                         hls::stream<ap_uint<SIZE_DWIDTH> >& dataSizeStream,
                         hls::stream<IntVectorStream_dt<OUT_DATAWIDTH, 1> >& outStream) {
    constexpr int16_t c_factor = IN_DATAWIDTH / OUT_DATAWIDTH;
    constexpr int16_t c_outWord = OUT_DATAWIDTH / 8;
    ap_uint<IN_DATAWIDTH> inVal;
    IntVectorStream_dt<OUT_DATAWIDTH, 1> outVal;
    ap_uint<SIZE_DWIDTH> inSize = 0;

downsizer_top:
    for (auto inSize = dataSizeStream.read(); inSize > 0; inSize = dataSizeStream.read()) {
        auto outSizeV = ((inSize - 1) / c_outWord) + 1;
        outVal.strobe = 1;
    downsizer_assign:
        for (ap_uint<SIZE_DWIDTH> dsize = 0; dsize < outSizeV; ++dsize) {
#pragma HLS PIPELINE II = 1
            auto idx = dsize % c_factor;
            if (idx == 0) {
                inVal = inStream.read();
            }
            outVal.data[0] = inVal.range(OUT_DATAWIDTH - 1, 0);
            inVal >>= OUT_DATAWIDTH;
            outStream << outVal;
        }
        // Block end Condition
        outVal.strobe = 0;
        outStream << outVal;
    }
    // File end Condition
    outVal.strobe = 0;
    outStream << outVal;
}

template <class SIZE_DT, int IN_WIDTH, int OUT_WIDTH>
void streamDownsizer(hls::stream<ap_uint<IN_WIDTH> >& inStream,
                     hls::stream<ap_uint<OUT_WIDTH> >& outStream,
                     SIZE_DT input_size) {
    /**
     * @brief This module reads the IN_WIDTH size from the data stream
     * and downsizes the data to OUT_WIDTH size and writes to output stream
     *
     * @tparam SIZE_DT data size
     * @tparam IN_WIDTH input width
     * @tparam OUT_WIDTH output width
     *
     * @param inStream input stream
     * @param outStream output stream
     * @param input_size input size
     */

    if (input_size == 0) // changed for gzip
        return;
    const int c_byteWidth = 8;
    const int c_inputWord = IN_WIDTH / c_byteWidth;
    const int c_outWord = OUT_WIDTH / c_byteWidth;
    uint32_t sizeOutputV = (input_size - 1) / c_outWord + 1;
    int factor = c_inputWord / c_outWord;
    ap_uint<IN_WIDTH> inBuffer = 0;
convInWidthtoV:
    for (int i = 0; i < sizeOutputV; i++) {
#pragma HLS PIPELINE II = 1
        int idx = i % factor;
        if (idx == 0) inBuffer = inStream.read();
        ap_uint<OUT_WIDTH> tmpValue = inBuffer;
        inBuffer >>= OUT_WIDTH;
        outStream << tmpValue;
    }
}

template <class SIZE_DT, int IN_WIDTH, int OUT_WIDTH>
void streamDownsizerP2P(hls::stream<ap_uint<IN_WIDTH> >& inStream,
                        hls::stream<ap_uint<OUT_WIDTH> >& outStream,
                        SIZE_DT input_size,
                        SIZE_DT input_start_idx) {
    /**
     * @brief This module reads the IN_WIDTH size from the data stream
     * and downsizes the data to OUT_WIDTH size and writes to output stream
     *
     * @tparam SIZE_DT data size
     * @tparam IN_WIDTH input width
     * @tparam OUT_WIDTH output width
     *
     * @param inStream input stream
     * @param outStream output stream
     * @param input_size input size
     * @param input_start_idx input starting index
     */
    const int c_byteWidth = 8;
    const int c_inputWord = IN_WIDTH / c_byteWidth;
    const int c_outWord = OUT_WIDTH / c_byteWidth;
    uint32_t sizeOutputV = (input_size - 1) / c_outWord + 1;
    int factor = c_inputWord / c_outWord;
    ap_uint<IN_WIDTH> inBuffer = 0;
    int offset = input_start_idx % c_inputWord;
convInWidthtoV:
    for (int i = offset; i < (sizeOutputV + offset); i++) {
#pragma HLS PIPELINE II = 1
        int idx = i % factor;
        if (idx == 0 || i == offset) inBuffer = inStream.read();
        ap_uint<OUT_WIDTH> tmpValue = inBuffer.range((idx + 1) * OUT_WIDTH - 1, idx * OUT_WIDTH);
        outStream << tmpValue;
    }
}

template <int IN_WIDTH, int PACK_WIDTH>
void streamDownSizerP2PComp(hls::stream<ap_uint<IN_WIDTH> >& inStream,
                            hls::stream<ap_uint<PACK_WIDTH> >& outStream,
                            hls::stream<uint32_t>& inStreamSize,
                            hls::stream<uint32_t>& outStreamSize,
                            uint32_t no_blocks) {
    const int c_byte_width = 8;
    const int c_input_word = IN_WIDTH / c_byte_width;
    const int c_out_word = PACK_WIDTH / c_byte_width;

    int factor = c_input_word / c_out_word;
    ap_uint<IN_WIDTH> inBuffer = 0;

    for (int size = inStreamSize.read(); size != 0; size = inStreamSize.read()) {
        // input size interms of 512width * 64 bytes after downsizing
        uint32_t sizeOutputV = (size - 1) / c_out_word + 1;

        // Send ouputSize of the module
        outStreamSize << size;

    // printf("[ %s ] sizeOutputV %d input_size %d size_4m_mm2s %d \n", __FUNCTION__, sizeOutputV, input_size, size);

    conv512toV:
        for (int i = 0; i < sizeOutputV; i++) {
#pragma HLS PIPELINE II = 1
            int idx = i % factor;
            if (idx == 0) inBuffer = inStream.read();
            ap_uint<PACK_WIDTH> tmpValue = inBuffer.range((idx + 1) * PACK_WIDTH - 1, idx * PACK_WIDTH);
            outStream << tmpValue;
        }
    }
}

} // namespace details
} // namespace compression
} // namespace xf

#endif // _XFCOMPRESSION_STREAM_DOWNSIZER_HPP_
