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
#ifndef _XFCOMPRESSION_LZX_DECOMPRESS_HPP_
#define _XFCOMPRESSION_LZX_DECOMPRESS_HPP_

/**
 * @file lz4_decompress.hpp
 * @brief Header for modules used in LZ4 decompression kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */
#include "hls_stream.h"
#include "lz_decompress.hpp"
#include "lz4_decompress.hpp"

#include <ap_int.h>
#include <assert.h>
#include <stdint.h>

namespace xf {
namespace compression {

template <int PARALLEL_BYTES, int HISTORY_SIZE>
void lzxDecompressEngine(hls::stream<ap_uint<PARALLEL_BYTES * 8> >& inStream,
                         hls::stream<ap_uint<PARALLEL_BYTES * 8> >& outStream,
                         hls::stream<bool>& outStreamEoS,
                         hls::stream<uint64_t>& outSizeStream,
                         const uint32_t _input_size) {
    typedef ap_uint<PARALLEL_BYTES * 8> uintV_t;
    typedef ap_uint<16> offset_dt;

    uint32_t input_size1 = _input_size;
    hls::stream<uint16_t> litlenStream("litlenStream");
    hls::stream<uintV_t> litStream("litStream");
    hls::stream<offset_dt> offsetStream("offsetStream");
    hls::stream<uint16_t> matchlenStream("matchlenStream");
#pragma HLS STREAM variable = litlenStream depth = 32
#pragma HLS STREAM variable = litStream depth = 32
#pragma HLS STREAM variable = offsetStream depth = 32
#pragma HLS STREAM variable = matchlenStream depth = 32

#pragma HLS BIND_STORAGE variable = litlenStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = litStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = offsetStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = matchlenStream type = FIFO impl = SRL

#pragma HLS dataflow
    lz4MultiByteDecompress<PARALLEL_BYTES>(inStream, litlenStream, litStream, offsetStream, matchlenStream,
                                           input_size1);
    lzMultiByteDecompress<PARALLEL_BYTES, HISTORY_SIZE>(litlenStream, litStream, offsetStream, matchlenStream,
                                                        outStream, outStreamEoS, outSizeStream);
}

} // namespace compression
} // namespace xf
#endif // _XFCOMPRESSION_LZX_DECOMPRESS_HPP_
