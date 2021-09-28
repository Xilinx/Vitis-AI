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
#include "kernel_mcbarrierbiasedengine.hpp"
#include "xf_fintech/mc_engine.hpp"
#include "xf_fintech/rng.hpp"
#include "xf_fintech/enums.hpp"
#include <ap_int.h>
#ifndef __SYNTHESIS__
#include <iostream>
#endif

extern "C" void McBarrierBiasedEngine_k(unsigned int loopNum,
                                        DtUsed underlying,
                                        DtUsed volatility,
                                        DtUsed dividendYield,
                                        DtUsed riskFreeRate,
                                        DtUsed timeLength, // model parameter
                                        DtUsed barrier,
                                        DtUsed strike,
                                        int optionType, // option parameter
                                        DtUsed out[OUTDEP],
                                        DtUsed rebate,
                                        DtUsed requiredTolerance,
                                        unsigned int requiredSamples,
                                        unsigned int timeSteps) {
#pragma HLS INTERFACE m_axi port = out bundle = gmem latency = 125
#pragma HLS INTERFACE s_axilite port = loopNum bundle = control
#pragma HLS INTERFACE s_axilite port = underlying bundle = control
#pragma HLS INTERFACE s_axilite port = volatility bundle = control
#pragma HLS INTERFACE s_axilite port = dividendYield bundle = control
#pragma HLS INTERFACE s_axilite port = riskFreeRate bundle = control
#pragma HLS INTERFACE s_axilite port = timeLength bundle = control
#pragma HLS INTERFACE s_axilite port = barrier bundle = control
#pragma HLS INTERFACE s_axilite port = strike bundle = control
#pragma HLS INTERFACE s_axilite port = optionType bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = rebate bundle = control
#pragma HLS INTERFACE s_axilite port = requiredTolerance bundle = control
#pragma HLS INTERFACE s_axilite port = requiredSamples bundle = control
#pragma HLS INTERFACE s_axilite port = timeSteps bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS data_pack variable = out
    ap_uint<32> seed[1];
    seed[0] = 5;

    // xf::fintech::enums::BarrierType barrierType = xf::fintech::enums::BarrierType::UpIn;
    ap_uint<2> barrierType = 2;

    bool optionTypeBool = optionType;

    for (int i = 0; i < loopNum; ++i) {
        xf::fintech::MCBarrierEngine<DtUsed, MCM_NM>(
            underlying, volatility, dividendYield, riskFreeRate, timeLength, barrier, strike, barrierType,
            optionTypeBool, // optionType,
            seed, &out[i], rebate, requiredTolerance, requiredSamples, timeSteps);
    }
}
