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

#include "m76.hpp"
#include "xf_fintech/m76_engine.hpp"
#include "xf_fintech/cf_bsm.hpp"

extern "C" {

void m76_kernel(struct xf::fintech::jump_diffusion_params<TEST_DT>* in, TEST_DT* out, int num_tests) {
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem_0
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem_1
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = num_tests bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS DATA_PACK variable = in

    struct xf::fintech::jump_diffusion_params<TEST_DT> local_in[MAX_NUMBER_TESTS];
    TEST_DT local_out[MAX_NUMBER_TESTS];
#pragma HLS ARRAY_PARTITION variable = local_in cyclic factor = 8 dim = 0
#pragma HLS DATA_PACK variable = local_in

/* copy all test vectors from global memory to local memory */
in_loop:
    for (int i = 0; i < num_tests; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1024 max = 2048 avg = 1024
        local_in[i] = in[i];
    }

    /* calculate the npv */
    TEST_DT sum_array[MAX_NUMBER_TESTS][MAX_N];
#pragma HLS ARRAY_PARTITION variable = sum_array cyclic factor = 8 dim = 0
#pragma HLS DATA_PACK variable = sum_array
calc_loop:
    for (int i = 0; i < num_tests / 8 + 1; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 128 max = 128 avg = 128
    inner_loop:
        for (int j = 0; j < 8; j++) {
#pragma HLS UNROLL
            int k = i * 8 + j;
            ;
            xf::fintech::M76Engine(&local_in[k], &sum_array[k][0]);
        }
    }

/* copy the results from local mem to global mem */
out_loop:
    for (int i = 0; i < num_tests; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1024 max = 2048 avg = 1024
        xf::fintech::internal::sum(&local_out[i], &sum_array[i][0]);
        out[i] = local_out[i];
    }
}

} // extern C
