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
#ifndef _XFCOMPRESSION_SNAPPY_DECOMPRESS_HPP_
#define _XFCOMPRESSION_SNAPPY_DECOMPRESS_HPP_

/**
 * @file snappy_decompress.hpp
 * @brief Header for modules used for snappy decompression kernle
 *
 * This file is part of Vitis Data Compression Library.
 */
#include "hls_stream.h"

#include <ap_int.h>
#include <assert.h>
#include <stdint.h>

namespace xf {
namespace compression {

template <typename T>
T reg(T d) {
#pragma HLS PIPELINE II = 1
#pragma HLS INTERFACE ap_ctrl_none port = return
#pragma HLS INLINE off
    return d;
}

typedef struct BlockInfo {
    ap_uint<17> compressSize;
    bool storedBlock;
} dt_blockInfo;

/**
 * @brief This module decodes the compressed data based on the snappy decompression format
 *
 * @param inStream input stream
 * @param outStream output stream
 * @param input_size input data size
 */
static void snappyDecompress(hls::stream<ap_uint<8> >& inStream,
                             hls::stream<ap_uint<32> >& outStream,
                             uint32_t input_size) {
    // Snappy Decoder states
    enum SnappyDecompressionStates {
        READ_STATE,
        MATCH_STATE,
        LOW_OFFSET_STATE,
        READ_TOKEN,
        READ_LITERAL,
        READ_LITLEN_60,
        READ_LITLEN_61,
        READ_OFFSET,
        READ_OFFSET_C01,
        READ_OFFSET_C10,
        READ_LITLEN_61_CONT,
        READ_OFFSET_C10_CONT
    };

    if (input_size == 0) return;

    enum SnappyDecompressionStates next_state = READ_TOKEN;
    ap_uint<8> nextValue;
    ap_uint<16> offset;
    ap_uint<32> decodedOut = 0;
    ap_uint<32> lit_len = 0;
    uint32_t match_len = 0;
    ap_uint<8> inValue = 0;
    bool read_instream = true;

    uint32_t inCntr_idx = 0;
    ap_uint<32> inBlkSize = 0;

    inValue = inStream.read();
    inCntr_idx++;

    if ((inValue >> 7) == 1) {
        inBlkSize.range(6, 0) = inValue.range(6, 0);
        inValue = inStream.read();
        inCntr_idx++;
        inBlkSize.range(13, 7) = inValue.range(6, 0);
        if ((inValue >> 7) == 1) {
            inValue = inStream.read();
            inCntr_idx++;
            inBlkSize.range(20, 14) = inValue.range(6, 0);
        }

    } else
        inBlkSize = inValue;

snappy_decompress:
    for (; inCntr_idx < input_size; inCntr_idx++) {
#pragma HLS PIPELINE II = 1
        if (read_instream)
            inValue = inStream.read();
        else
            inCntr_idx--;

        read_instream = true;
        if (next_state == READ_TOKEN) {
            if (inValue.range(1, 0) != 0) {
                next_state = READ_OFFSET;
                read_instream = false;
            } else {
                lit_len = inValue.range(7, 2);

                if (lit_len < 60) {
                    lit_len++;
                    next_state = READ_LITERAL;
                } else if (lit_len == 60) {
                    next_state = READ_LITLEN_60;
                } else if (lit_len == 61) {
                    next_state = READ_LITLEN_61;
                }
            }
        } else if (next_state == READ_LITERAL) {
            ap_uint<32> outValue = 0;
            outValue.range(7, 0) = inValue;
            outStream << outValue;
            lit_len--;
            if (lit_len == 0) next_state = READ_TOKEN;

        } else if (next_state == READ_OFFSET) {
            offset = 0;
            if (inValue.range(1, 0) == 1) {
                match_len = inValue.range(4, 2);
                offset.range(10, 8) = inValue.range(7, 5);
                next_state = READ_OFFSET_C01;
            } else if (inValue.range(1, 0) == 2) {
                match_len = inValue.range(7, 2);
                next_state = READ_OFFSET_C10;
            } else {
                next_state = READ_TOKEN;
                read_instream = false;
            }
        } else if (next_state == READ_OFFSET_C01) {
            offset.range(7, 0) = inValue;
            ap_uint<32> outValue = 0;
            outValue.range(31, 16) = match_len + 3;
            outValue.range(15, 0) = offset - 1;
            outStream << outValue;
            next_state = READ_TOKEN;
        } else if (next_state == READ_OFFSET_C10) {
            offset.range(7, 0) = inValue;
            next_state = READ_OFFSET_C10_CONT;
        } else if (next_state == READ_OFFSET_C10_CONT) {
            offset.range(15, 8) = inValue;
            ap_uint<32> outValue = 0;

            outValue.range(31, 16) = match_len;
            outValue.range(15, 0) = offset - 1;
            outStream << outValue;
            next_state = READ_TOKEN;

        } else if (next_state == READ_LITLEN_60) {
            lit_len = inValue + 1;
            next_state = READ_LITERAL;
        } else if (next_state == READ_LITLEN_61) {
            lit_len.range(7, 0) = inValue;
            next_state = READ_LITLEN_61_CONT;
        } else if (next_state == READ_LITLEN_61_CONT) {
            lit_len.range(15, 8) = inValue;
            lit_len = lit_len + 1;
            next_state = READ_LITERAL;
        }
    } // End of main snappy_decoder for-loop
}

template <int PARALLEL_BYTES, class SIZE_DT = ap_uint<17>, bool BLOCKD_ONLY = 0>
static void snappyMultiByteDecompress(hls::stream<ap_uint<PARALLEL_BYTES * 8> >& inStream,
                                      hls::stream<SIZE_DT>& litlenStream,
                                      hls::stream<ap_uint<PARALLEL_BYTES * 8> >& litStream,
                                      hls::stream<ap_uint<16> >& offsetStream,
                                      hls::stream<SIZE_DT>& matchlenStream,
                                      hls::stream<dt_blockInfo>& blockInfoStream) {
    // Snappy Decoder states
    enum SnappyDecompressionStates { READ_TOKEN, READ_LITERAL };

    for (dt_blockInfo bInfo = blockInfoStream.read(); bInfo.compressSize != 0; bInfo = blockInfoStream.read()) {
        SIZE_DT input_size = bInfo.compressSize;
        enum SnappyDecompressionStates next_state = READ_TOKEN;
        ap_uint<16> offset;
        ap_uint<32> decodedOut = 0;
        SIZE_DT lit_len = 0;
        SIZE_DT match_len = 0;
        ap_uint<8> inValue = 0, inValue1 = 0;

        SIZE_DT readBytes = 0, processedBytes = 0;
        uint8_t inputIdx = 0;

        const int c_parallelBit = PARALLEL_BYTES * 8;
        ap_uint<3 * c_parallelBit> input_window;
        ap_uint<c_parallelBit> outStreamValue;

        for (int i = 0; i < 3; i++) {
#pragma HLS PIPELINE II = 1
            input_window.range((i + 1) * c_parallelBit - 1, i * c_parallelBit) = inStream.read();
            readBytes += PARALLEL_BYTES;
        }

        uint8_t incrIdx = 2 * PARALLEL_BYTES;
        bool storedBlock = bInfo.storedBlock;

        if (storedBlock) {
            lit_len = input_size;
            litlenStream << lit_len;
            next_state = READ_LITERAL;
        }

        if (BLOCKD_ONLY) {
            ap_uint<32> inBlkSize = 0;
            inValue = input_window.range((inputIdx + 1) * 8 - 1, inputIdx * 8);
            inputIdx++;

            inValue1 = input_window.range((inputIdx + 1) * 8 - 1, inputIdx * 8);
            inputIdx++;

            bool flag = false;
            bool c0 = ((inValue >> 7) == 1);
            bool c1 = ((inValue1 >> 7) == 1);
            if (c0 & c1) {
                inBlkSize.range(6, 0) = inValue.range(6, 0);
                inBlkSize.range(13, 7) = inValue1.range(6, 0);
                ap_uint<8> inValue2 = input_window.range((inputIdx + 1) * 8 - 1, inputIdx * 8);
                inputIdx++;
                inBlkSize.range(20, 14) = inValue2.range(6, 0);
                processedBytes += 3;
            } else if (c0) {
                inBlkSize.range(6, 0) = inValue.range(6, 0);
                inBlkSize.range(13, 7) = inValue1.range(6, 0);
                processedBytes += 2;
            } else {
                inBlkSize = inValue;
                inputIdx--;
                processedBytes += 1;
            }

            incrIdx -= inputIdx;
            input_window >>= inputIdx * 8;
            inputIdx = 0;
        }

    snappyMultiByteDecompress:
        for (; processedBytes + inputIdx < input_size;) {
#pragma HLS PIPELINE II = 1
            uint8_t incrInputIdx = 0;
            if (next_state == READ_TOKEN) {
                ap_uint<24> inValue = input_window >> (inputIdx * 8);
                if (inValue.range(1, 0) != 0) {
                    offset = 0;
                    offset.range(7, 0) = inValue(15, 8);
                    if (inValue.range(1, 0) == 1) {
                        match_len = inValue.range(4, 2);
                        offset.range(10, 8) = inValue.range(7, 5);
                        incrInputIdx = 2;

                        match_len += 4;

                    } else if (inValue.range(1, 0) == 2) {
                        match_len = inValue.range(7, 2);
                        offset.range(15, 8) = inValue.range(23, 16);
                        incrInputIdx = 3;

                        match_len += 1;
                    }
                    next_state = READ_TOKEN;
                    litlenStream << 0;
                    offsetStream << offset;
                    matchlenStream << match_len;
                } else {
                    ap_uint<6> localLitLen = inValue.range(7, 2);

                    if (localLitLen == 60) {
                        lit_len = inValue.range(15, 8) + 1;
                        incrInputIdx = 2;
                    } else if (localLitLen == 61) {
                        lit_len = inValue.range(23, 8) + 1;
                        incrInputIdx = 3;
                    } else {
                        lit_len = localLitLen + 1;
                        incrInputIdx = 1;
                    }
                    litlenStream << lit_len;
                    next_state = READ_LITERAL;
                }
            } else if (next_state == READ_LITERAL) {
                litStream << (input_window >> (inputIdx * 8));
                if (lit_len >= PARALLEL_BYTES) {
                    incrInputIdx = PARALLEL_BYTES;
                    lit_len -= PARALLEL_BYTES;
                } else {
                    incrInputIdx = lit_len;
                    lit_len = 0;
                }
                if (lit_len == 0) {
                    next_state = READ_TOKEN;
                    matchlenStream << 0;
                    offsetStream << 0;
                } else {
                    next_state = READ_LITERAL;
                }
            } else {
                assert(0);
            }

            inputIdx += incrInputIdx;
            bool inputIdxFlag = reg<bool>((inputIdx >= PARALLEL_BYTES));
            // write to input stream based on PARALLEL BYTES
            if (inputIdxFlag) {
                input_window >>= c_parallelBit;
                inputIdx -= PARALLEL_BYTES;
                processedBytes += PARALLEL_BYTES;
                if (readBytes < input_size) {
                    ap_uint<c_parallelBit> input = inStream.read();
                    readBytes += PARALLEL_BYTES;
                    if (BLOCKD_ONLY) {
                        input_window.range((incrIdx + PARALLEL_BYTES) * 8 - 1, incrIdx * 8) = input;
                    } else {
                        input_window.range(3 * c_parallelBit - 1, 2 * c_parallelBit) = input;
                    }
                }
            }

        } // End of main snappy_decoder for-loop
    }
    // signalling end of transaction
    litlenStream << 0;
    matchlenStream << 0;
    offsetStream << 0;
}

} // namespace compression
} // namespace xf

#endif // _XFCOMPRESSION_SNAPPY_DECOMPRESS_HPP_
