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
#ifndef _XFCOMPRESSION_BLOCK_COMP_DM_HPP_
#define _XFCOMPRESSION_BLOCK_COMP_DM_HPP_

/**
 * @file block_comp_datamover_kernel.hpp
 * @brief Header for data mover kernel which streams data to compression
 * streaming kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include "axi_stream_utils.hpp"
#include "mm2s.hpp"
#include "s2mm.hpp"
#include "stream_downsizer.hpp"
#include "stream_upsizer.hpp"

#include "ap_axi_sdata.h"
#include "hls_stream.h"

extern "C" {
/**
 * @brief Data mover kernel top function for block based compression algorithms.
 * It reads
 *        data from memory and streams it to block compression kernel.
 *
 * @param in input stream
 * @param out output stream
 * @param compressed_size decompressed size output
 * @param input_size input size
 * @param instream_orig input axi kernel stream (written by this kernel)
 * @param blocksizestream_orig input axi kernel stream (written by this kernel)
 * @param outstream_dest output axi kernel stream (read by this kernel)
 *
 */
void xilCompDatamover(xf::compression::uintMemWidth_t* in,
                      xf::compression::uintMemWidth_t* out,
                      uint32_t* compressed_size,
                      uint32_t input_size,
                      hls::stream<ap_axiu<8, 0, 0, 0> >& instream_orig,
                      hls::stream<ap_axiu<8, 0, 0, 0> >& outstream_dest);
}
#endif // _XFCOMPRESSION_BLOCK_COMP_DM_HPP_
