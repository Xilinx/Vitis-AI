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
 * @file lz4_packer_mm.cpp
 * @brief Source for LZ4 P2P compression kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */
#include "lz4_packer_mm.hpp"

extern "C" {
void xilLz4Packer(uint512_t* in,
                  uint512_t* out,
                  uint32_t* compressd_size,
                  uint32_t* in_block_size,
                  uint32_t* encoded_size,
                  uint512_t* orig_input_data,
                  uint32_t block_size_in_kb,
                  uint32_t no_blocks,
                  uint32_t xxhashVal,
                  uint32_t input_size) {
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = compressd_size offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = in_block_size offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = encoded_size offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = orig_input_data offset = slave bundle = gmem
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = compressd_size bundle = control
#pragma HLS INTERFACE s_axilite port = in_block_size bundle = control
#pragma HLS INTERFACE s_axilite port = encoded_size bundle = control
#pragma HLS INTERFACE s_axilite port = orig_input_data bundle = control
#pragma HLS INTERFACE s_axilite port = block_size_in_kb bundle = control
#pragma HLS INTERFACE s_axilite port = no_blocks bundle = control
#pragma HLS INTERFACE s_axilite port = xxhashVal bundle = control
#pragma HLS INTERFACE s_axilite port = input_size bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    xf::compression::lz4PackerMM<GMEM_DWIDTH, GMEM_BURST_SIZE>(orig_input_data, in, out, in_block_size, compressd_size,
                                                               encoded_size, xxhashVal, block_size_in_kb, no_blocks,
                                                               input_size);
    return;
}
}
