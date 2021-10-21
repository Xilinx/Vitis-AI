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
// top header file
#include "xf_fintech/mc_engine.hpp"
#include "xf_fintech/rng.hpp"
#include "kernel_MCAsianGPEngine.hpp"

extern "C" void kernel_MCAsianGP_0(TEST_DT underlying,
                                   TEST_DT volatility,
                                   TEST_DT dividendYield,
                                   TEST_DT riskFreeRate, // model parameter
                                   TEST_DT timeLength,
                                   TEST_DT strike,
                                   int optionType, // option parameter
                                   TEST_DT outputs[1],
                                   TEST_DT requiredTolerance,
                                   unsigned int requiredSamples,
                                   unsigned int timeSteps,
                                   unsigned int maxSamples) {
#pragma HLS INTERFACE m_axi port = outputs bundle = gmem latency = 125

#pragma HLS INTERFACE s_axilite port = underlying bundle = control
#pragma HLS INTERFACE s_axilite port = volatility bundle = control
#pragma HLS INTERFACE s_axilite port = dividendYield bundle = control
#pragma HLS INTERFACE s_axilite port = riskFreeRate bundle = control
#pragma HLS INTERFACE s_axilite port = timeLength bundle = control
#pragma HLS INTERFACE s_axilite port = strike bundle = control
#pragma HLS INTERFACE s_axilite port = optionType bundle = control
#pragma HLS INTERFACE s_axilite port = outputs bundle = control
#pragma HLS INTERFACE s_axilite port = requiredTolerance bundle = control
#pragma HLS INTERFACE s_axilite port = requiredSamples bundle = control
#pragma HLS INTERFACE s_axilite port = timeSteps bundle = control
#pragma HLS INTERFACE s_axilite port = maxSamples bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    const int MCM_NM = 4;
    bool option = (optionType == 1) ? 1 : 0;

    ap_uint<32> seed[MCM_NM];
    for (int i = 0; i < MCM_NM; ++i) {
        seed[i] = 5000 + i * 1000;
    }

    xf::fintech::MCAsianGeometricAPEngine<TEST_DT, MCM_NM>(underlying, volatility, dividendYield, riskFreeRate,
                                                           timeLength, strike, option, seed, outputs, requiredTolerance,
                                                           requiredSamples, timeSteps, maxSamples);
}
