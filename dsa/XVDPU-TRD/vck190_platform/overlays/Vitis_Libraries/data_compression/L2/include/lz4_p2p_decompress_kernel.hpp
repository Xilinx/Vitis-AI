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
#ifndef _XFCOMPRESSION_LZ4_P2P_DECOMPRESS_KERNEL_HPP_
#define _XFCOMPRESSION_LZ4_P2P_DECOMPRESS_KERNEL_HPP_

/**
 * @file lz4_p2p_decompress_kernel.hpp
 * @brief Header for LZ4 P2P decompression kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "hls_stream.h"
#include <ap_int.h>

#include "lz_decompress.hpp"
#include "mm2s.hpp"
#include "s2mm.hpp"
#include "stream_downsizer.hpp"
#include "stream_upsizer.hpp"
#include "lz4_decompress.hpp"
#include "lz4_p2p.hpp"
#define GMEM_DWIDTH 512
#define GMEM_BURST_SIZE 16

// Kernel top functions
extern "C" {

/**
 * @brief LZ4 P2P decompression kernel is responsible for decompressing data
 * which is in LZ4 encoded form.
 *
 * @param in input stream width
 * @param out output stream width
 * @param in_block_size input size
 * @param in_compress_size output size
 * @param block_start_idx start index of block
 * @param no_blocks number of blocks for each compute unit
 * @param block_size_in_kb block input size
 * @param compute_unit particular compute unit
 * @param total_no_cu number of compute units
 * @param num_blocks number of blocks base don host buffersize
 */
void xilLz4P2PDecompress(const xf::compression::uintMemWidth_t* in,
                         xf::compression::uintMemWidth_t* out,
                         dt_blockInfo* bObj,
                         dt_chunkInfo* cObj,
                         uint32_t block_size_in_kb,
                         uint32_t compute_unit,
                         uint8_t total_no_cu,
                         uint32_t num_blocks);
}

#endif // _XFCOMPRESSION_LZ4_P2P_DECOMPRESS_KERNEL_HPP_
