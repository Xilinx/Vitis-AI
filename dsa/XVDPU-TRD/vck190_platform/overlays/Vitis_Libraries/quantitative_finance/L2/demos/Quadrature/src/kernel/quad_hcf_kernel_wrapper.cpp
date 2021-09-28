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

#include "quad_hcf_engine_def.hpp"

extern "C" {

void quad_hcf_kernel(struct hcfEngineInputDataType* in, TEST_DT* out, int num_tests) {
#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem_0
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem_1
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = num_tests bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS DATA_PACK variable = in

    /* copy all test vectors from global memory to local memory */
    struct hcfEngineInputDataType local_in[MAX_NUMBER_TESTS];
    TEST_DT local_out[MAX_NUMBER_TESTS];

    for (int i = 0; i < num_tests; i++) {
        local_in[i] = in[i];
    }

    /* calculate the npv */
    for (int i = 0; i < num_tests; i++) {
        local_out[i] = xf::fintech::hcfEngine(&local_in[i]);
    }

    /* copy the results from local mem to global mem */
    for (int i = 0; i < num_tests; i++) {
        out[i] = local_out[i];
    }
}

} // extern C
