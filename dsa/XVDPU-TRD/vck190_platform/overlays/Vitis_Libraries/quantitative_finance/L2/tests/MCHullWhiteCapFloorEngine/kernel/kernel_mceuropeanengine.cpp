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
                            DtUsed nomial,
                            DtUsed initRate,
                            DtUsed strike,
                            bool isCap,
                            DtUsed singlePeriod,
                            DtUsed alpha,
                            DtUsed sigma,
                            DtUsed output[OUTDEP],
                            DtUsed requiredTolerance,
                            unsigned int requiredSamples,
                            unsigned int timeSteps) {
#pragma HLS INTERFACE m_axi port = output bundle = gmem latency = 125

#pragma HLS INTERFACE s_axilite port = loop_nm bundle = control
#pragma HLS INTERFACE s_axilite port = nomial bundle = control
#pragma HLS INTERFACE s_axilite port = initRate bundle = control
#pragma HLS INTERFACE s_axilite port = strike bundle = control
#pragma HLS INTERFACE s_axilite port = isCap bundle = control
#pragma HLS INTERFACE s_axilite port = singlePeriod bundle = control
#pragma HLS INTERFACE s_axilite port = alpha bundle = control
#pragma HLS INTERFACE s_axilite port = sigma bundle = control
#pragma HLS INTERFACE s_axilite port = output bundle = control
#pragma HLS INTERFACE s_axilite port = requiredTolerance bundle = control
#pragma HLS INTERFACE s_axilite port = requiredSamples bundle = control
#pragma HLS INTERFACE s_axilite port = timeSteps bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS data_pack variable = output

    ap_uint<32> seed[MCM_NM];
    for (int i = 0; i < MCM_NM; ++i) {
        seed[i] = 12 + i * 1000;
    }

    for (int i = 0; i < loop_nm; ++i) {
        xf::fintech::MCHullWhiteCapFloorEngine<DtUsed, MCM_NM>(nomial, initRate, strike, isCap, singlePeriod, alpha,
                                                               sigma, seed, output, requiredTolerance, requiredSamples,
                                                               timeSteps);
    }
}
