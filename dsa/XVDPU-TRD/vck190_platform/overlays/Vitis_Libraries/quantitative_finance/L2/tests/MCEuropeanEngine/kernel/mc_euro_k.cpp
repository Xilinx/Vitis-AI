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
#include "mc_euro_k.hpp"
#ifndef __SYNTHESIS__
#include <iostream>
#endif

extern "C" void mc_euro_k(TEST_DT underlying,
                          TEST_DT volatility,
                          TEST_DT dividendYield,
                          TEST_DT riskFreeRate, // model parameter
                          TEST_DT timeLength,
                          TEST_DT strike,
                          unsigned int optionType, // option parameter
                          ap_uint<32> seed[2],
                          TEST_DT output[1],
                          TEST_DT requiredTolerance,
                          unsigned int requiredSamples,
                          unsigned int timeSteps) {
#pragma HLS INTERFACE m_axi port = output bundle = gmem0 offset = slave
#pragma HLS INTERFACE m_axi port = seed bundle = gmem1 offset = slave

#pragma HLS INTERFACE s_axilite port = underlying bundle = control
#pragma HLS INTERFACE s_axilite port = volatility bundle = control
#pragma HLS INTERFACE s_axilite port = dividendYield bundle = control
#pragma HLS INTERFACE s_axilite port = riskFreeRate bundle = control
#pragma HLS INTERFACE s_axilite port = timeLength bundle = control
#pragma HLS INTERFACE s_axilite port = strike bundle = control
#pragma HLS INTERFACE s_axilite port = optionType bundle = control
#pragma HLS INTERFACE s_axilite port = seed bundle = control
#pragma HLS INTERFACE s_axilite port = output bundle = control
#pragma HLS INTERFACE s_axilite port = requiredTolerance bundle = control
#pragma HLS INTERFACE s_axilite port = requiredSamples bundle = control
#pragma HLS INTERFACE s_axilite port = timeSteps bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    xf::fintech::MCEuropeanEngine<TEST_DT, 2>(underlying, volatility, dividendYield,
                                              riskFreeRate, // model parameter
                                              timeLength, strike,
                                              optionType, // option parameter
                                              seed, output, requiredTolerance, requiredSamples, timeSteps);
}
