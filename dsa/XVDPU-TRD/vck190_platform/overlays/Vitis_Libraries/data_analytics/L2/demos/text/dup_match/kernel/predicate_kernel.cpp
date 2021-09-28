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

#include "predicate_kernel.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

extern "C" void TGP_Kernel(ConfigParam config,
                           uint8_t* field,
                           uint32_t* offset,
                           double* idfValue,
                           uint64_t* tfAddr,
                           AXI_DT* tfValue,
                           AXI_DT* tfValue2,
                           AXI_DT* tfValue3,
                           AXI_DT* tfValue4,
                           uint32_t* indexId) {
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 32 bundle = gmem0 port = field depth = 4096
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 32 bundle = gmem1 port = offset depth = 64
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 32 bundle = gmem2 port = idfValue depth = 4096
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 32 bundle = gmem3 port = tfAddr depth = 4096
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 32 bundle = gmem4 port = tfValue depth = 131072
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 32 bundle = gmem5 port = tfValue2 depth = 131072
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 32 bundle = gmem6 port = tfValue3 depth = 131072
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 1 num_read_outstanding = \
    16 max_write_burst_length = 2 max_read_burst_length = 32 bundle = gmem7 port = tfValue4 depth = 131072
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 num_read_outstanding = \
    1 max_write_burst_length = 32 max_read_burst_length = 2 bundle = gmem8 port = indexId depth = 64

#pragma HLS INTERFACE s_axilite port = config bundle = control
#pragma HLS INTERFACE s_axilite port = field bundle = control
#pragma HLS INTERFACE s_axilite port = offset bundle = control
#pragma HLS INTERFACE s_axilite port = idfValue bundle = control
#pragma HLS INTERFACE s_axilite port = tfAddr bundle = control
#pragma HLS INTERFACE s_axilite port = tfValue bundle = control
#pragma HLS INTERFACE s_axilite port = tfValue2 bundle = control
#pragma HLS INTERFACE s_axilite port = tfValue3 bundle = control
#pragma HLS INTERFACE s_axilite port = tfValue4 bundle = control
#pragma HLS INTERFACE s_axilite port = indexId bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#ifndef __SYNTHESIS__
    std::cout << "Start Predicate Kernel" << std::endl;
#endif
    xf::data_analytics::text::TwoGramPredicate<AXI_DT> pred;
    pred.init(config.docSize, config.fldSize, idfValue, tfAddr);
    pred.search(field, offset, tfValue, tfValue2, tfValue3, tfValue4, indexId);
#ifndef __SYNTHESIS__
    std::cout << "End Predicate Kernel" << std::endl;
#endif
}
