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
#include "hjm_kernel.hpp"

using TEST_PRICER = xf::fintech::hjmZeroCouponBondPricer<TEST_DT, TEST_MAX_TENORS>;

extern "C" void hjm_kernel(TEST_DT* historicalData,
                           unsigned noTenors,
                           unsigned noCurves,
                           float simYears,
                           unsigned noPaths,
                           float zcbMaturity,
                           unsigned* mcSeeds,
                           TEST_DT* outputPrice) {
#pragma HLS INTERFACE m_axi port = historicalData bundle = data_in
#pragma HLS INTERFACE m_axi port = outputPrice bundle = data_out
#pragma HLS INTERFACE m_axi port = mcSeeds bundle = data_seeds

#pragma HLS INTERFACE s_axilite port = historicalData bundle = control
#pragma HLS INTERFACE s_axilite port = noTenors bundle = control
#pragma HLS INTERFACE s_axilite port = noCurves bundle = control
#pragma HLS INTERFACE s_axilite port = noPaths bundle = control
#pragma HLS INTERFACE s_axilite port = outputPrice bundle = control
#pragma HLS INTERFACE s_axilite port = mcSeeds bundle = control
#pragma HLS INTERFACE s_axilite port = simYears bundle = control
#pragma HLS INTERFACE s_axilite port = zcbMaturity bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS DATAFLOW

    hls::stream<TEST_DT> priceOut("price_out_strm");

    ap_uint<32> seedsArr[TEST_MC_UN][xf::fintech::hjmModelParams::N];
#pragma HLS ARRAY_PARTITION variable = seedsArr complete
    for (unsigned i = 0; i < TEST_MC_UN; i++) {
        for (unsigned j = 0; j < xf::fintech::hjmModelParams::N; j++) {
#pragma HLS PIPELINE
            seedsArr[i][j] = mcSeeds[i * xf::fintech::hjmModelParams::N + j];
        }
    }

    TEST_PRICER pricer[TEST_MC_UN][1];
#pragma HLS ARRAY_PARTITION variable = pricer dim = 1 complete
    for (unsigned i = 0; i < TEST_MC_UN; i++) {
#pragma HLS UNROLL
        pricer[i][0].init(noTenors, zcbMaturity);
    }

    xf::fintech::hjmEngine<TEST_DT, TEST_PRICER, TEST_MAX_TENORS, TEST_MAX_CURVES, TEST_PCA_NCU, TEST_MC_UN>(
        noTenors, noCurves, simYears, noPaths, historicalData, pricer, seedsArr, priceOut);
    *outputPrice = priceOut.read();
}
