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
 * @brief HWA Option Pricing Engine
 * @param a The mean reversion
 * @param sigma The volatility
 * @param times Vector of maturity's
 * @param rates Vector of interest rates
 * @param t Vector of current time t
 * @param T Vector of bond maturity time T
 * @param S Vector of option maturity time S
 * @param K Vector of strike prices K
 * @param P Output vector of bond prices
 */

extern "C" void HWA_k1(TEST_DT a,
                       TEST_DT sigma,
                       TEST_DT times[LEN],
                       TEST_DT rates[LEN],
                       int types[N_k1],
                       TEST_DT t[N_k1],
                       TEST_DT T[N_k1],
                       TEST_DT S[N_k1],
                       TEST_DT K[N_k1],
                       TEST_DT P[N_k1]) {
#ifndef HLS_TEST
#pragma HLS INTERFACE m_axi port = times offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = rates offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = types offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = t offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = T offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = S offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = K offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = P offset = slave bundle = gmem1

#pragma HLS INTERFACE s_axilite port = a bundle = control
#pragma HLS INTERFACE s_axilite port = sigma bundle = control
#pragma HLS INTERFACE s_axilite port = times bundle = control
#pragma HLS INTERFACE s_axilite port = rates bundle = control
#pragma HLS INTERFACE s_axilite port = types bundle = control
#pragma HLS INTERFACE s_axilite port = t bundle = control
#pragma HLS INTERFACE s_axilite port = T bundle = control
#pragma HLS INTERFACE s_axilite port = S bundle = control
#pragma HLS INTERFACE s_axilite port = K bundle = control
#pragma HLS INTERFACE s_axilite port = P bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#endif

    TEST_DT local_times[LEN];
    TEST_DT local_rates[LEN];
    int local_types[N_k1];
    TEST_DT local_t[N_k1];
    TEST_DT local_T[N_k1];
    TEST_DT local_S[N_k1];
    TEST_DT local_K[N_k1];
#pragma HLS array_partition variable = local_times complete dim = 1
#pragma HLS array_partition variable = local_rates complete dim = 1
#pragma HLS array_partition variable = local_t complete dim = 1
#pragma HLS array_partition variable = local_T complete dim = 1
#pragma HLS array_partition variable = local_S complete dim = 1
#pragma HLS array_partition variable = local_K complete dim = 1

// copy to local memory from global
loop_hwa_k1_0:
    for (int i = 0; i < LEN; i++) {
#pragma HLS unroll
        local_times[i] = times[i];
        local_rates[i] = rates[i];
    }

loop_hwa_k1_1:
    for (int i = 0; i < N_k1; i++) {
#pragma HLS unroll
        local_types[i] = types[i];
        local_t[i] = t[i];
        local_T[i] = T[i];
        local_S[i] = S[i];
        local_K[i] = K[i];
    }

    // create instance of the engine
    HWAEngine<TEST_DT, LEN> hwaEngine;
    hwaEngine.init(a, sigma, local_times, local_rates);

// loop around and calculate all the option prices
loop_hwa_k1_2:
    for (int i = 0; i < N_k1; i++) {
        P[i] = hwaEngine.optionPrice(local_types[i], local_t[i], local_T[i], local_S[i], local_K[i]);

#ifndef __SYNTHESIS__
        std::cout << i << " t-->T:" << local_t[i] << ":" << local_T[i];
        std::cout << " S:" << local_S[i] << " K:" << local_K[i] << std::endl;
        std::cout << " Option Price: " << P[i] << std::endl;
#endif
    }
}
