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

extern "C" void kernel_mc_0(unsigned int loop_nm,
                            DtUsed underlying,
                            DtUsed volatility,
                            DtUsed dividendYield,
                            DtUsed riskFreeRate, // model parameter
                            DtUsed timeLength,
                            DtUsed strike,
                            bool optionType, // option parameter
                            DtUsed out[OUTDEP],
                            DtUsed requiredTolerance,
                            unsigned int requiredSamples,
                            unsigned int timeSteps,
                            unsigned int maxSamples) {
#pragma HLS INTERFACE m_axi port = out bundle = gmem latency = 125

#pragma HLS INTERFACE s_axilite port = loop_nm bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = strike bundle = control
#pragma HLS INTERFACE s_axilite port = underlying bundle = control
#pragma HLS INTERFACE s_axilite port = volatility bundle = control
#pragma HLS INTERFACE s_axilite port = dividendYield bundle = control
#pragma HLS INTERFACE s_axilite port = riskFreeRate bundle = control
#pragma HLS INTERFACE s_axilite port = timeLength bundle = control
#pragma HLS INTERFACE s_axilite port = requiredTolerance bundle = control
#pragma HLS INTERFACE s_axilite port = requiredSamples bundle = control
#pragma HLS INTERFACE s_axilite port = timeSteps bundle = control
#pragma HLS INTERFACE s_axilite port = maxSamples bundle = control
#pragma HLS INTERFACE s_axilite port = optionType bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS data_pack variable = out

    ap_uint<32> seed[MCM_NM];
    for (int i = 0; i < MCM_NM; ++i) {
        seed[i] = 4332 + i * 1000;
    }
    for (int i = 0; i < loop_nm; ++i) {
        xf::fintech::MCEuropeanPriBypassEngine<DtUsed, MCM_NM>(
            underlying, volatility, dividendYield, riskFreeRate, timeLength, strike, optionType, seed, &out[i],
            requiredTolerance, requiredSamples, timeSteps, maxSamples);
    }
}
