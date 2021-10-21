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

#ifndef _XFCOMPRESSION_GZIP_DM_S2MM_HPP_
#define _XFCOMPRESSION_GZIP_DM_S2MM_HPP_

/**
 * @file gzip_dm_s2mm.hpp
 * @brief Header for gzip datamover s2mm.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include "ap_axi_sdata.h"

#include "hls_stream.h"
#include "s2mm.hpp"
#include "zlib_specs.hpp"
#include <ap_int.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

typedef ap_uint<MULTIPLE_BYTES * 8> uintMemWidth_t;

extern "C" {
/**
 * @brief Gzip data mover:S2MM kernel top function.
 *
 * @param out output memory pointer
 * @param encoded_size decompressed size output
 * @param read_block_size Block size to be read
 * @param intream input axi stream (512-bit wide data stream read by this
 * kernel)
 *
 */

void xilGzipS2MM(uintMemWidth_t* out,
                 uint32_t* encoded_size,
                 uint32_t* status_flag,
                 uint32_t read_block_size,
                 hls::stream<ap_axiu<MULTIPLE_BYTES * 8, 0, 0, 0> >& inStream);
}
#endif // _XFCOMPRESSION_GZIP_DM_S2MM_HPP_
