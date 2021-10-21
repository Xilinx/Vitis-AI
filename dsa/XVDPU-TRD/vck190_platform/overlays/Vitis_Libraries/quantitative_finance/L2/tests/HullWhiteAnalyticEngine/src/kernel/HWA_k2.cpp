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
 * @brief HWA Cap/Floor Engine Pricing
 * @param a The mean reversion
 * @param sigma The volatility
 * @param times Vector of maturity's
 * @param rates Vector of interest rates
 * @param capfloorType Cap or Floor
 * @param startYear Vector of Start Years
 * @param endYear Vector of End Years
 * @param settlementFreq The settlement frequency (1,2 or 4)
 * @param N Vector of Nomial values
 * @param X Vector rates
 * @param P Output vector of calculated prices
 */

extern "C" void HWA_k2(TEST_DT a,
                       TEST_DT sigma,
                       TEST_DT times[LEN],
                       TEST_DT rates[LEN],
                       int capfloorType[N_k2],
                       TEST_DT startYear[N_k2],
                       TEST_DT endYear[N_k2],
                       int settlementFreq[N_k2],
                       TEST_DT N[N_k2],
                       TEST_DT X[N_k2],
                       TEST_DT P[N_k2]) {
#ifndef HLS_TEST
#pragma HLS INTERFACE m_axi port = times offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = rates offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = capfloorType offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = startYear offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = endYear offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = settlementFreq offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = N offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = X offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = P offset = slave bundle = gmem2

#pragma HLS INTERFACE s_axilite port = a bundle = control
#pragma HLS INTERFACE s_axilite port = sigma bundle = control
#pragma HLS INTERFACE s_axilite port = times bundle = control
#pragma HLS INTERFACE s_axilite port = rates bundle = control
#pragma HLS INTERFACE s_axilite port = capfloorType bundle = control
#pragma HLS INTERFACE s_axilite port = startYear bundle = control
#pragma HLS INTERFACE s_axilite port = endYear bundle = control
#pragma HLS INTERFACE s_axilite port = settlementFreq bundle = control
#pragma HLS INTERFACE s_axilite port = N bundle = control
#pragma HLS INTERFACE s_axilite port = X bundle = control
#pragma HLS INTERFACE s_axilite port = P bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#endif

    TEST_DT local_times[LEN];
    TEST_DT local_rates[LEN];
    int local_capfloorType[N_k2];
    TEST_DT local_startYear[N_k2];
    TEST_DT local_endYear[N_k2];
    int local_settlementFreq[N_k2];
    TEST_DT local_N[N_k2];
    TEST_DT local_X[N_k2];
#pragma HLS array_partition variable = local_times complete dim = 1
#pragma HLS array_partition variable = local_rates complete dim = 1
#pragma HLS array_partition variable = local_capfloorType complete dim = 1
#pragma HLS array_partition variable = local_startYear complete dim = 1
#pragma HLS array_partition variable = local_endYear complete dim = 1
#pragma HLS array_partition variable = local_settlementFreq complete dim = 1
#pragma HLS array_partition variable = local_N complete dim = 1
#pragma HLS array_partition variable = local_X complete dim = 1

// copy to local memory from global
loop_hwa_k2_0:
    for (int i = 0; i < LEN; i++) {
#pragma HLS unroll
        local_times[i] = times[i];
        local_rates[i] = rates[i];
    }

loop_hwa_k2_1:
    for (int i = 0; i < N_k2; i++) {
#pragma HLS unroll
        local_capfloorType[i] = capfloorType[i];
        local_startYear[i] = startYear[i];
        local_endYear[i] = endYear[i];
        local_settlementFreq[i] = settlementFreq[i];
        local_N[i] = N[i];
        local_X[i] = X[i];
    }

    // create instance of the engine
    HWAEngine<TEST_DT, LEN> hwaEngine;
    hwaEngine.init(a, sigma, local_times, local_rates);

loop_hwa_k2_2:
    for (int i = 0; i < N_k2; i++) {
        // calculate the cap or floor
        P[i] = hwaEngine.capfloorPrice(local_capfloorType[i], local_startYear[i], local_endYear[i],
                                       local_settlementFreq[i], local_N[i], local_X[i]);

#ifndef __SYNTHESIS__
        std::cout << i << " Cap/Floor Price:" << P[i] << std::endl;
#endif
    }
}
