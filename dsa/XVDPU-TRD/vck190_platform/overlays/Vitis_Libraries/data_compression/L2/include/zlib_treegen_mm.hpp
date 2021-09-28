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

#ifndef _XFCOMPRESSION_ZLIB_TREEGEN_MM_HPP_
#define _XFCOMPRESSION_ZLIB_TREEGEN_MM_HPP_

/**
 * @file zlib_treegen_mm.hpp
 * @brief Header for tree generator kernel used in zlib compression.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include "zlib_specs.hpp"

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include <ap_int.h>
#include "huffman_treegen.hpp"

extern "C" {
/**
 * @brief This is a resource optimized version of huffman treegen kernel.
 * It takes literal and distance frequency data as input through single input stream
 * and generates dynamic huffman codes and bit length data which is output through a single output stream.
 * This kernel does not use DDR in any way and is optimised for both speed and low resource usage.
 *
 * @param freqStream 24-bit input stream for getting frequency data
 * @param codeStream 20-bit output stream sending huffman codes and bit-lengths data
 *
 */
/*void xilTreegenKernel(hls::stream<ap_axiu<c_frequency_bits, 0, 0, 0> > &freqStream,
                      hls::stream<ap_axiu<c_codeword_bits, 0, 0, 0> > &codeStream);
*/
void xilTreegenKernel(uint32_t* dyn_ltree_freq,
                      uint32_t* dyn_dtree_freq,
                      uint32_t* dyn_bltree_freq,
                      uint32_t* dyn_ltree_codes,
                      uint32_t* dyn_dtree_codes,
                      uint32_t* dyn_bltree_codes,
                      uint32_t* dyn_ltree_blen,
                      uint32_t* dyn_dtree_blen,
                      uint32_t* dyn_bltree_blen,
                      uint32_t* max_codes,
                      uint32_t block_size_in_kb,
                      uint32_t input_size,
                      uint32_t blocks_per_chunk);
}

#endif // _XFCOMPRESSION_ZLIB_TREEGEN_MM_HPP_
