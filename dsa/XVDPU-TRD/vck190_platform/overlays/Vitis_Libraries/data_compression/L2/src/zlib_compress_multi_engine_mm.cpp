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
/**
 * @file zlib_compress_multi_engine_mm.cpp
 * @brief Source for Zlib compression kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include "zlib_compress_multi_engine_mm.hpp"

extern "C" {
/**
 * @brief Zlib compression kernel.
 *
 * @param in input stream width
 * @param out output stream width
 * @param compressd_size output size
 * @param input_size input size
 */
void xilZlibCompressFull

    (const ap_uint<GMEM_DWIDTH>* in, ap_uint<GMEM_DWIDTH>* out, uint32_t* compressd_size, uint32_t input_size) {
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = compressd_size offset = slave bundle = gmem
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = compressd_size bundle = control
#pragma HLS INTERFACE s_axilite port = input_size bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS dataflow
    xf::compression::zlibCompressMultiEngineMM<GMEM_DWIDTH, PARALLEL_BLOCK, GMEM_BURST_SIZE, ZLIB_BLOCK_SIZE>(
        in, out, compressd_size, input_size);
}
}
