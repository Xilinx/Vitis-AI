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

extern "C" void kernel_mc_0(unsigned int loop_num,
                            DtUsed underlying[asset_nm],
                            DtUsed riskFreeRate,
                            DtUsed sigma[asset_nm],
                            DtUsed v0[asset_nm],
                            DtUsed theta[asset_nm],
                            DtUsed kappa[asset_nm],
                            DtUsed rho[asset_nm],
                            DtUsed dividendYield[asset_nm],
                            bool optionType,
                            DtUsed strike,
                            DtUsed timeLength,
                            DtUsed outputs[OUTDEP],
                            DtUsed requiredTolerance,
                            unsigned int requiredSamples,
                            unsigned int timeSteps,
                            unsigned int maxSamples) {
#pragma HLS INTERFACE m_axi port = underlying bundle = gmem0 latency = 125
#pragma HLS INTERFACE m_axi port = sigma bundle = gmem1 latency = 125
#pragma HLS INTERFACE m_axi port = v0 bundle = gmem2 latency = 125
#pragma HLS INTERFACE m_axi port = theta bundle = gmem3 latency = 125
#pragma HLS INTERFACE m_axi port = kappa bundle = gmem4 latency = 125
#pragma HLS INTERFACE m_axi port = rho bundle = gmem5 latency = 125
#pragma HLS INTERFACE m_axi port = dividendYield bundle = gmem6 latency = 125
#pragma HLS INTERFACE m_axi port = outputs bundle = gmem7 latency = 125

#pragma HLS INTERFACE s_axilite port = loop_num bundle = control
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
#pragma HLS INTERFACE s_axilite port = outputs bundle = control
#pragma HLS INTERFACE s_axilite port = requiredTolerance bundle = control
#pragma HLS INTERFACE s_axilite port = requiredSamples bundle = control
#pragma HLS INTERFACE s_axilite port = timeSteps bundle = control
#pragma HLS INTERFACE s_axilite port = maxSamples bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS data_pack variable = underlying
#pragma HLS data_pack variable = sigma
#pragma HLS data_pack variable = v0
#pragma HLS data_pack variable = theta
#pragma HLS data_pack variable = kappa
#pragma HLS data_pack variable = rho
#pragma HLS data_pack variable = dividendYield
#pragma HLS data_pack variable = outputs

    ap_uint<32> seed[MCM_NM][2];
    for (int i = 0; i < MCM_NM; ++i) {
        for (int j = 0; j < 2; j++) {
            seed[i][j] = i * 10000 + j * 227 + 1;
        }
    }
    DtUsed corrMatrix[asset_nm * 2 + 1][asset_nm];
    for (int i = 0; i < asset_nm * 2 + 1; i++) {
        for (int j = 0; j < asset_nm; j++) {
            if (i == 0 || i == asset_nm * 2 - j) {
                corrMatrix[i][j] = 1.0;
            } else {
                corrMatrix[i][j] = 0.0;
            }
        }
    }

    for (int i = 0; i < loop_num; ++i) {
        xf::fintech::MCMultiAssetEuropeanHestonEngine<DtUsed, asset_nm, MCM_NM, xf::fintech::kDTQuadraticExponential>(
            underlying, riskFreeRate, sigma, v0, theta, kappa, corrMatrix, rho, dividendYield, optionType, strike,
            timeLength, seed, outputs, requiredTolerance, requiredSamples, timeSteps, maxSamples);
    }
}
