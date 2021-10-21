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
#include "re_engine_kernel.hpp"

extern "C" void reEngineKernel(ap_uint<64>* cfg_buff,
                               ap_uint<64>* msg_buff,
                               ap_uint<16>* len_buff,
                               ap_uint<32>* out_buff) {
#pragma HLS interface m_axi offset = slave latency = 64 num_read_outstanding = 2 max_read_burst_length = 16 bundle = \
    gmem0_0 port = cfg_buff depth = 1000
#pragma HLS interface s_axilite port = cfg_buff bundle = control

#pragma HLS interface m_axi offset = slave latency = 64 num_read_outstanding = 2 max_read_burst_length = 16 bundle = \
    gmem0_0 port = msg_buff depth = 30000
#pragma HLS interface s_axilite port = msg_buff bundle = control

#pragma HLS interface m_axi offset = slave latency = 64 num_read_outstanding = 2 max_read_burst_length = 16 bundle = \
    gmem0_1 port = len_buff depth = 2000
#pragma HLS interface s_axilite port = len_buff bundle = control

#pragma HLS interface m_axi offset = slave latency = 64 num_write_outstanding = 2 max_read_burst_length = 16 bundle = \
    gmem1_0 port = out_buff depth = 20000
#pragma HLS interface s_axilite port = out_buff bundle = control

#pragma HLS interface s_axilite port = return bundle = control

    xf::data_analytics::text::reEngine<PU_NM, INSTR_DEPTH, CCLASS_NM, CPGP_NM, MSG_LEN, STACK_SIZE>(cfg_buff, msg_buff,
                                                                                                    len_buff, out_buff);
}
