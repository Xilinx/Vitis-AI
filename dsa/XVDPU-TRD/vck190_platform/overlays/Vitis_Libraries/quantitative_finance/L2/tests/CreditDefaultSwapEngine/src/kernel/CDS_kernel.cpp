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

#include "cds_engine_kernel.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

/**
 * @brief TODO
 * @param
 */

extern "C" void CDS_kernel(TEST_DT timesIR[IRLEN],
                           TEST_DT ratesIR[IRLEN],
                           TEST_DT timesHazard[HAZARDLEN],
                           TEST_DT ratesHazard[HAZARDLEN],
                           TEST_DT notional[N],
                           TEST_DT recovery[N],
                           TEST_DT maturity[N],
                           int frequency[N],
                           TEST_DT cds[N]) {
#ifndef HLS_TEST
#pragma HLS INTERFACE m_axi port = timesIR offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = ratesIR offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = timesHazard offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = ratesHazard offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = notional offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = recovery offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = maturity offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = frequency offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = cds offset = slave bundle = gmem0

#pragma HLS INTERFACE s_axilite port = cds bundle = control
#pragma HLS INTERFACE s_axilite port = frequency bundle = control
#pragma HLS INTERFACE s_axilite port = maturity bundle = control
#pragma HLS INTERFACE s_axilite port = recovery bundle = control
#pragma HLS INTERFACE s_axilite port = notional bundle = control
#pragma HLS INTERFACE s_axilite port = ratesHazard bundle = control
#pragma HLS INTERFACE s_axilite port = timesHazard bundle = control
#pragma HLS INTERFACE s_axilite port = ratesIR bundle = control
#pragma HLS INTERFACE s_axilite port = timesIR bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#endif

    TEST_DT local_timesIR[IRLEN];
    TEST_DT local_ratesIR[IRLEN];
    TEST_DT local_timesHazard[HAZARDLEN];
    TEST_DT local_ratesHazard[HAZARDLEN];
    TEST_DT local_notional[N];
    TEST_DT local_recovery[N];
    TEST_DT local_maturity[N];
    int local_frequency[N];
    TEST_DT local_cds[N];

#pragma HLS array_partition variable = local_timesIR complete dim = 1
#pragma HLS array_partition variable = local_ratesIR complete dim = 1
#pragma HLS array_partition variable = local_timesHazard complete dim = 1
#pragma HLS array_partition variable = local_ratesHazard complete dim = 1
#pragma HLS array_partition variable = local_notional complete dim = 1
#pragma HLS array_partition variable = local_recovery complete dim = 1
#pragma HLS array_partition variable = local_maturity complete dim = 1
#pragma HLS array_partition variable = local_frequency complete dim = 1
#pragma HLS array_partition variable = local_cds complete dim = 1

// copy to local memory from global
loop_cds_ker_0:
    for (int i = 0; i < IRLEN; i++) {
#pragma HLS unroll
        local_timesIR[i] = timesIR[i];
        local_ratesIR[i] = ratesIR[i];
    }

loop_cds_ker_1:
    for (int i = 0; i < HAZARDLEN; i++) {
#pragma HLS unroll
        local_timesHazard[i] = timesHazard[i];
        local_ratesHazard[i] = ratesHazard[i];
    }

loop_cds_ker_2:
    for (int i = 0; i < N; i++) {
#pragma HLS unroll
        local_notional[i] = notional[i];
        local_recovery[i] = recovery[i];
        local_maturity[i] = maturity[i];
        local_frequency[i] = frequency[i];
    }

    // create instance of the engine
    CDSEngine<TEST_DT, IRLEN, HAZARDLEN> cdsEngine;
    cdsEngine.init(local_timesIR, local_ratesIR, local_timesHazard, local_ratesHazard);

loop_cds_ker_3:
    for (int i = 0; i < N; i++) {
        local_cds[i] = cdsEngine.cdsSpread(local_frequency[i], local_maturity[i], local_recovery[i]);
#ifndef __SYNTHESIS__
        std::cout << i << " Freq:" << local_frequency[i] << std::endl;
        std::cout << i << " Maturity:" << local_maturity[i] << std::endl;
        std::cout << i << " Recovery:" << local_recovery[i] << std::endl;
        std::cout << i << " CDS Spread:" << local_cds[i] << std::endl;
#endif
    }

loop_cds_ker_4:
    for (int i = 0; i < N; i++) {
#pragma HLS unroll
        cds[i] = local_cds[i];
    }
}
