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

#include "xf_data_analytics/regression/linearRegressionTrain.hpp"

extern "C" void ridgeRegressionTrain(ap_uint<512>* input, ap_uint<512>* output) {
// clang-format off
#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
    num_write_outstanding = 2 num_read_outstanding = 32 \
    max_write_burst_length = 2 max_read_burst_length = 32 \
    bundle = gmem0_0 port = input

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
    num_write_outstanding = 32 num_read_outstanding = 2 \
    max_write_burst_length = 32 max_read_burst_length = 2 \
    bundle = gmem0_1 port = output

#pragma HLS INTERFACE s_axilite port = input bundle = control
#pragma HLS INTERFACE s_axilite port = output bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
    // clang-format on
    xf::data_analytics::regression::ridgeRegressionSGDTrain<512, 8, 25, 64>(input, output);
}
