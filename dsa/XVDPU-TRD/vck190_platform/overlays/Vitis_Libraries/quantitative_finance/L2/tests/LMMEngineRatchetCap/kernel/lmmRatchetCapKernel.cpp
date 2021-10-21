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
/**
 * @brief Kernel wrapper for Libor Maket Model Ratchet Cap pricer
 */
#include "lmmRatchetCapKernel.hpp"

extern "C" void lmmRatchetCapKernel(unsigned noTenors,
                                    unsigned noPaths,
                                    TEST_DT* presentRate,
                                    TEST_DT rhoBeta,
                                    TEST_DT* capletVolas,
                                    TEST_DT notional,
                                    TEST_DT spread,
                                    TEST_DT kappa0,
                                    ap_uint<32>* seeds,
                                    TEST_DT* outPrice) {
#pragma HLS INTERFACE m_axi port = presentRate bundle = in_rate
#pragma HLS INTERFACE m_axi port = seeds bundle = in_seeds
#pragma HLS INTERFACE m_axi port = capletVolas bundle = in_capletVolas
#pragma HLS INTERFACE m_axi port = outPrice bundle = out_price

#pragma HLS INTERFACE s_axilite port = noTenors bundle = control
#pragma HLS INTERFACE s_axilite port = noPaths bundle = control
#pragma HLS INTERFACE s_axilite port = presentRate bundle = control
#pragma HLS INTERFACE s_axilite port = rhoBeta bundle = control
#pragma HLS INTERFACE s_axilite port = capletVolas bundle = control
#pragma HLS INTERFACE s_axilite port = notional bundle = control
#pragma HLS INTERFACE s_axilite port = spread bundle = control
#pragma HLS INTERFACE s_axilite port = kappa0 bundle = control
#pragma HLS INTERFACE s_axilite port = seeds bundle = control
#pragma HLS INTERFACE s_axilite port = outPrice bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS DATAFLOW

    hls::stream<TEST_DT> sigmaStrm("sigmaStrm"), rhoStrm("rhoStrm");

    xf::fintech::lmmCorrelationGenerator<TEST_DT>::oneParametricCorrelation(rhoBeta, noTenors, rhoStrm);
    xf::fintech::lmmVolatilityGenerator<TEST_DT, TEST_MAX_TENORS>::piecewiseConstVolatility(noTenors, capletVolas,
                                                                                            sigmaStrm);

    TEST_PT pricer[TEST_UN][1];
#pragma HLS ARRAY_PARTITION variable = pricer complete
    for (unsigned i = 0; i < TEST_UN; i++) {
#pragma HLS UNROLL
        pricer[i][0].init(notional, spread, kappa0);
    }

    ap_uint<32> seedsArr[TEST_UN];
#pragma HLS ARRAY_PARTITION variable = seedsArr complete
    for (unsigned i = 0; i < TEST_UN; i++) {
#pragma HLS PIPELINE
        seedsArr[i] = seeds[i];
    }

    xf::fintech::lmmEngine<TEST_DT, TEST_PT, TEST_MAX_TENORS, TEST_NF, TEST_UN, TEST_PCA_UN>(
        noTenors, noPaths, rhoStrm, presentRate, sigmaStrm, pricer, seedsArr, outPrice);
}
