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
 * @file crc32_mm.cpp
 * @brief Source for Crc32 Kernel.
 *
 * This file is part of Vitis Data Compression Library.
 */

#include "crc32_mm.hpp"

extern "C" {
/**
 * @brief Crc32 Kernel.
 */
void xilCrc32

    (const ap_uint<PARALLEL_BYTES * 8>* in, ap_uint<32>* crcData, uint32_t inSize) {
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = crcData offset = slave bundle = gmem
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = crcData bundle = control
#pragma HLS INTERFACE s_axilite port = inSize bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    xf::compression::crc32_mm<PARALLEL_BYTES>(in, crcData, inSize);
}
}
