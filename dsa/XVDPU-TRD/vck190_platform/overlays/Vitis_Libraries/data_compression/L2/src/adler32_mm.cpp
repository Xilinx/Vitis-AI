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
 * @file adler32_mm.cpp
 * @brief Source for Adler32 Kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include "adler32_mm.hpp"

extern "C" {
/**
 * @brief Adler32 Kernel.
 */
void xilAdler32

    (const ap_uint<PARALLEL_BYTES * 8>* in, ap_uint<32>* adlerData, uint32_t inSize) {
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = adlerData offset = slave bundle = gmem
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = adlerData bundle = control
#pragma HLS INTERFACE s_axilite port = inSize bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    xf::compression::adler32_mm<PARALLEL_BYTES>(in, adlerData, inSize);
}
}
