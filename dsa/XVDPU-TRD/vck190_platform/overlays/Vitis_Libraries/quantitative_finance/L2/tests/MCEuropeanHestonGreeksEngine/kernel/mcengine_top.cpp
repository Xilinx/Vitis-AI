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
#include "mcengine_top.hpp"
extern "C" void MCEHGEngine_k0(TEST_DT underlying,
                               TEST_DT riskFreeRate,
                               TEST_DT sigma,
                               TEST_DT v0,
                               TEST_DT theta,
                               TEST_DT kappa,
                               TEST_DT rho,
                               TEST_DT dividendYield,
                               unsigned int optionType,
                               TEST_DT strike,
                               TEST_DT timeLength,
                               unsigned int timeSteps,
                               unsigned int requiredSamples,
                               unsigned int maxSamples,
                               ap_uint<32> seed[8 * 2],
                               TEST_DT requiredTolerance,
                               TEST_DT outputs[8]) {
#pragma HLS INTERFACE m_axi port = seed bundle = gmem0 offset = slave
#pragma HLS INTERFACE m_axi port = outputs bundle = gmem1 offset = slave

#pragma HLS INTERFACE s_axilite port = underlying bundle = control
#pragma HLS INTERFACE s_axilite port = riskFreeRate bundle = control
#pragma HLS INTERFACE s_axilite port = sigma bundle = control
#pragma HLS INTERFACE s_axilite port = v0 bundle = control
#pragma HLS INTERFACE s_axilite port = theta bundle = control
#pragma HLS INTERFACE s_axilite port = kappa bundle = control
#pragma HLS INTERFACE s_axilite port = rho bundle = control
#pragma HLS INTERFACE s_axilite port = dividendYield bundle = control
#pragma HLS INTERFACE s_axilite port = optionType bundle = control
#pragma HLS INTERFACE s_axilite port = strike bundle = control
#pragma HLS INTERFACE s_axilite port = timeLength bundle = control
#pragma HLS INTERFACE s_axilite port = timeSteps bundle = control
#pragma HLS INTERFACE s_axilite port = requiredSamples bundle = control
#pragma HLS INTERFACE s_axilite port = maxSamples bundle = control
#pragma HLS INTERFACE s_axilite port = seed bundle = control
#pragma HLS INTERFACE s_axilite port = requiredTolerance bundle = control
#pragma HLS INTERFACE s_axilite port = outputs bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
    ap_uint<32> seed_2d[8][2];
    for (unsigned int i = 0; i < 8; i++) {
        for (unsigned int j = 0; j < 2; j++) {
            seed_2d[i][j] = seed[i * 2 + j];
        }
    }
    bool optionTypeIn = optionType;
    xf::fintech::MCEuropeanHestonGreeksEngine(underlying, riskFreeRate, sigma, v0, theta, kappa, rho, dividendYield,
                                              optionTypeIn, strike, timeLength, seed_2d, outputs, requiredTolerance,
                                              requiredSamples, timeSteps, maxSamples);
}
