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
#include "dut.hpp"

void dut(ap_uint<WDATA> ii, hls::stream<ap_uint<WDATA> >& out_stream) {
    xf::common::utils_hw::UramArray<WDATA, NDATA, NCACHE> uram_array1;

l_init_value:
    int num = uram_array1.memSet(ii);

l_read_after_write_test:
    for (int i = 0; i < NUM_SIZE; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = uram_array1.blocks inter false
        if ((i & 1) == 0) {
            uram_array1.write(i, i);
        } else {
            ap_uint<WDATA> t = uram_array1.read(i - 1);
            out_stream.write(t);
        }
    }

// test case requires WData > 36, otherwise cosim will fail
l_update_value_with_1_II:
    for (int i = 0; i < NUM_SIZE; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = uram_array1.blocks inter false
        ap_uint<WDATA> t = uram_array1.read(i);
        ap_uint<WDATA> u = (t & 1) ? 1 : 0;
        uram_array1.write(i, u);
    }

l_dump_value:
    for (int i = 0; i < NDATA; ++i) {
#pragma HLS PIPELINE II = 1
        ap_uint<WDATA> t = uram_array1.read(i);
        out_stream.write(t);
    }
}
