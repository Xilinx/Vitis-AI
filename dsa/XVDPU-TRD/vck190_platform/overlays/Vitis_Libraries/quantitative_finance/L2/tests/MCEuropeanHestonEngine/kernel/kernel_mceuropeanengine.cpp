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
#include "kernel_mceuropeanengine.hpp"
#include "xf_fintech/mc_engine.hpp"
#include "xf_fintech/rng.hpp"
#include "xf_fintech/enums.hpp"

extern "C" void kernel_mc_0(unsigned int loop_nm,
                            DtUsed underlying,
                            DtUsed riskFreeRate,
                            DtUsed sigma,
                            DtUsed v0,
                            DtUsed theta,
                            DtUsed kappa,
                            DtUsed rho,
                            DtUsed dividendYield,
                            bool optionType,
                            DtUsed strike,
                            DtUsed timeLength,
                            unsigned int timeSteps,
                            unsigned int requiredSamples,
                            unsigned int maxSamples,
                            DtUsed requiredTolerance,
                            DtUsed outputs[OUTDEP]) {
#pragma HLS INTERFACE m_axi port = outputs bundle = gmem latency = 125

#pragma HLS INTERFACE s_axilite port = loop_nm bundle = control
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
#pragma HLS INTERFACE s_axilite port = requiredTolerance bundle = control
#pragma HLS INTERFACE s_axilite port = outputs bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS data_pack variable = outputs

    ap_uint<32> seed[MCM_NM][2];
    for (int i = 0; i < MCM_NM; ++i) {
        for (int j = 0; j < 2; j++) {
            seed[i][j] = i * 10000 + j * 227 + 1;
        }
    }
    for (int i = 0; i < loop_nm; ++i) {
        xf::fintech::MCEuropeanHestonEngine<DtUsed, MCM_NM, xf::fintech::kDTQuadraticExponential, false>(
            underlying, riskFreeRate, sigma, v0, theta, kappa, rho, dividendYield, optionType, strike, timeLength, seed,
            outputs, requiredTolerance, requiredSamples, timeSteps, maxSamples);
    }
}
