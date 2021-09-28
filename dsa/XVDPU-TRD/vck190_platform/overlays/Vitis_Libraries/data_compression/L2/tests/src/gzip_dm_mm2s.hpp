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

#ifndef _XFCOMPRESSION_GZIP_DM_MM2S_HPP_
#define _XFCOMPRESSION_GZIP_DM_MM2S_HPP_

/**
 * @file gzip_dm_m2s.hpp
 * @brief Header for gzip datamover mm2s.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include "ap_axi_sdata.h"

#include "hls_stream.h"
#include "mm2s.hpp"
#include "stream_downsizer.hpp"
#include "zlib_specs.hpp"
#include <ap_int.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

typedef ap_uint<MULTIPLE_BYTES * 8> uintMemWidth_t;

#ifdef INZSTD
const int c_inStreamDwidth = (MULTIPLE_BYTES * 8);
#else
const int c_inStreamDwidth = 16;
#endif

extern "C" {
/**
 * @brief Gzip data mover:Data Mover kernel top function. It reads data from
 * memory and streams it
 *  to gzip decompression kernel.
 *
 * @param in input memory pointer
 * @param inputSize input size
 * @param last flag for last set of data
 * @param outStream AXI Stream output
 *
 */

void xilGzipMM2S(uintMemWidth_t* in,
                 uint32_t inputSize,
                 uint32_t last,
                 hls::stream<ap_axiu<c_inStreamDwidth, 0, 0, 0> >& outStream);
}
#endif // _XFCOMPRESSION_GZIP_DM_MM2S_HPP_
