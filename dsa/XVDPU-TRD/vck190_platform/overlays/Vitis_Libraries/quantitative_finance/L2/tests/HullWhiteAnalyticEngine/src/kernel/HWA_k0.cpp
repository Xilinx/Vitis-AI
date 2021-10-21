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

#include "hwa_engine_kernel.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

/**
 * @brief HWA Bond Pricing Engine
 * @param a The mean reversion
 * @param sigma The volatility
 * @param times A vector of maturity's
 * @param rates A vector of interest rates
 * @param t A vector of current time t
 * @param T A vector of maturity time T
 * @param P  Output vector of bond prices
 */

extern "C" void HWA_k0(TEST_DT a,
                       TEST_DT sigma,
                       TEST_DT times[LEN],
                       TEST_DT rates[LEN],
                       TEST_DT t[N_k0],
                       TEST_DT T[N_k0],
                       TEST_DT P[N_k0]) {
#ifndef HLS_TEST
#pragma HLS INTERFACE m_axi port = times offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = rates offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = t offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = T offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = P offset = slave bundle = gmem0

#pragma HLS INTERFACE s_axilite port = a bundle = control
#pragma HLS INTERFACE s_axilite port = sigma bundle = control
#pragma HLS INTERFACE s_axilite port = times bundle = control
#pragma HLS INTERFACE s_axilite port = rates bundle = control
#pragma HLS INTERFACE s_axilite port = t bundle = control
#pragma HLS INTERFACE s_axilite port = T bundle = control
#pragma HLS INTERFACE s_axilite port = P bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#endif

    TEST_DT local_times[LEN];
    TEST_DT local_rates[LEN];
    TEST_DT local_t[N_k0];
    TEST_DT local_T[N_k0];
#pragma HLS array_partition variable = local_times complete dim = 1
#pragma HLS array_partition variable = local_rates complete dim = 1
#pragma HLS array_partition variable = local_t complete dim = 1
#pragma HLS array_partition variable = local_T complete dim = 1

// copy to local memory from global
loop_hwa_k0_0:
    for (int i = 0; i < LEN; i++) {
#pragma HLS unroll
        local_times[i] = times[i];
        local_rates[i] = rates[i];
    }

loop_hwa_k0_1:
    for (int i = 0; i < N_k0; i++) {
#pragma HLS unroll
        local_t[i] = t[i];
        local_T[i] = T[i];
    }

    // create instance of the engine
    HWAEngine<TEST_DT, LEN> hwaEngine;
    hwaEngine.init(a, sigma, local_times, local_rates);

loop_hwa_k0_2:
    for (int i = 0; i < N_k0; i++) {
        P[i] = hwaEngine.bondPrice(local_t[i], local_T[i]);
#ifndef __SYNTHESIS__
        std::cout << i << " BondPrice:" << P[i] << std::endl;
#endif
    }
}
