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

#ifndef _XFCOMPRESSION_GZIP_BLOCK_COMP_DM_HPP_
#define _XFCOMPRESSION_GZIP_BLOCK_COMP_DM_HPP_

/**
 * @file gzip_block_comp_dm.hpp
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
#include <ap_int.h>

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#ifndef GMEM_IN_DWIDTH
#define GMEM_IN_DWIDTH 8
#endif

#ifndef GMEM_OUT_DWIDTH
#define GMEM_OUT_DWIDTH 16
#endif

extern "C" {
/**
 * @brief Data mover kernel top function for decompression kernel
 * implementations.
 *        It reads data from memory and streams it to target kernel.
 *
 * @param in                input stream
 * @param out               output stream
 * @param input_size        input size
 * @param outputSize        decompressed size output
 * @param instream_orig     input axi kernel stream (written by this kernel)
 * @param instream_size     input axi size stream (written by this kernel)
 * @param outstream_dest    output axi kernel stream (read by this kernel)
 *
 */
void xilCompDatamover(xf::compression::uintMemWidth_t* in,
                      xf::compression::uintMemWidth_t* out,
                      uint32_t input_size,
                      uint32_t* outputSize,
                      hls::stream<ap_axiu<GMEM_IN_DWIDTH, 0, 0, 0> >& instream_orig,
                      hls::stream<ap_axiu<32, 0, 0, 0> >& instream_size,
                      hls::stream<ap_axiu<GMEM_OUT_DWIDTH, 0, 0, 0> >& outstream_dest);
}
#endif // _XFCOMPRESSION_GZIP_BLOCK_DECOMP_DM_HPP_
