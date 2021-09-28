/*
 * (c) Copyright 2021 Xilinx, Inc. All rights reserved.
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

#ifndef _XFCOMPRESSION_ZSTD_DATAMOVER_HPP_
#define _XFCOMPRESSION_ZSTD_DATAMOVER_HPP_

/**
 * @file zstd_datamover.hpp
 * @brief Header for data mover kernel which streams data to compression
 * streaming kernel
 *        and moves the streamed out to DDR.
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

#ifndef STREAM_IN_DWIDTH
#define STREAM_IN_DWIDTH 8
#endif

#ifndef STREAM_OUT_DWIDTH
#define STREAM_OUT_DWIDTH 32
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
 * @param outputSize        output data size
 * @param origStream        input axi kernel stream (written by this kernel)
 * @param destStream        output axi kernel stream (read by this kernel)
 *
 */
void xilZstdDataMover(xf::compression::uintMemWidth_t* in,
                      xf::compression::uintMemWidth_t* out,
                      uint32_t input_size,
                      uint32_t* outputSize,
                      hls::stream<ap_axiu<STREAM_IN_DWIDTH, 0, 0, 0> >& origStream,
                      hls::stream<ap_axiu<STREAM_OUT_DWIDTH, 0, 0, 0> >& destStream);
}
#endif // _XFCOMPRESSION_ZSTD_DATAMOVER_HPP_
