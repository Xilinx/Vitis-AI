/*
 * Copyright 2019 Xilinx, Inc.
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
 */

/**
 *
 * @file aes256CbcEncryptKernel.cpp
 * @brief kernel code of Cipher Block Chaining (CBC) block cipher mode of operation.
 * This file is part of Vitis Security Library.
 *
 * @detail Containing scan, distribute, encrypt, merge, and write-out functions.
 *
 */

#include "xf_data_analytics/clustering/kmeansTrain.hpp"
#include <ap_int.h>
#include <hls_stream.h>
#include "config.hpp"

// @brief top of kernel
extern "C" void kmeansKernel(ap_uint<512>* inputData, ap_uint<512>* centers) {
// clang-format off
#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 4 num_read_outstanding = 32 \
	max_write_burst_length = 4 max_read_burst_length = 64 \
	bundle = gmem0_0 port = inputData 

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 16 num_read_outstanding = 16 \
	max_write_burst_length = 64 max_read_burst_length = 64 \
	bundle = gmem0_1 port = centers 

#pragma HLS INTERFACE s_axilite port = inputData bundle = control
#pragma HLS INTERFACE s_axilite port = centers bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    // clang-format on

    xf::data_analytics::clustering::kMeansTrain<DType, DIM, KC, PCU, PDV>(inputData, centers);
}
