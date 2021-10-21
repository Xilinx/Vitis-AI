/*
 * Copyright 2021 Xilinx, Inc.
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

#include "merge_sort_kernel.hpp"
#include "common.hpp"

extern "C" void gqe_merge_kernel_0(ap_uint<2 * (DATA_WIDTH + KEY_WIDTH)>* inBuff,
                                   unsigned int eachLen,
                                   unsigned int totalLen,
                                   unsigned int order,
                                   unsigned int outBuffOff,
                                   ap_uint<2 * (DATA_WIDTH + KEY_WIDTH)>* outBuff) {
    enum { buff_Depth = 4 * INSERT_LEN * URAM_NUM * CH_NUM * CH_NUM * CH_NUM * CH_NUM_1 / 2 };
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 max_write_burst_length = \
    128 bundle = gmem0_0 port = inBuff depth = buff_Depth
#pragma HLS INTERFACE s_axilite port = inBuff bundle = control

#pragma HLS INTERFACE s_axilite port = eachLen bundle = control
#pragma HLS INTERFACE s_axilite port = totalLen bundle = control
#pragma HLS INTERFACE s_axilite port = order bundle = control
#pragma HLS INTERFACE s_axilite port = outBuffOff bundle = control

#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 16 max_write_burst_length = \
    128 bundle = gmem0_1 port = outBuff depth = buff_Depth
#pragma HLS INTERFACE s_axilite port = outBuff bundle = control

#pragma HLS interface s_axilite port = return bundle = control

    xf::database::mergeSortKernel<ap_uint<DATA_WIDTH>, ap_uint<KEY_WIDTH>, CH_NUM, BURST_LEN, DATA_WIDTH, KEY_WIDTH>(
        inBuff, eachLen, totalLen, order, outBuffOff, outBuff);
}
