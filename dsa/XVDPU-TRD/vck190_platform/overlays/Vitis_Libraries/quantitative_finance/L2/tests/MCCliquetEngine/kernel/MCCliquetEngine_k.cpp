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
#ifndef __SYNTHESIS__
#include <iostream>
#endif

extern "C" void MCCliquetEngine_k(TEST_DT underlying,
                                  TEST_DT volatility,
                                  TEST_DT dividendYield,
                                  TEST_DT riskFreeRate,
                                  TEST_DT timeLength, // Process
                                  TEST_DT strike,
                                  int optionTypeIn,
                                  TEST_DT resetDates[1024],
                                  ap_uint<32>* seed,
                                  TEST_DT* output,
                                  TEST_DT requiredTolerance,
                                  unsigned int timeSteps,
                                  unsigned int requiredSamples) {
#pragma HLS INTERFACE m_axi port = output bundle = gmem0 offset = slave
#pragma HLS INTERFACE m_axi port = resetDates bundle = gmem1 offset = slave
#pragma HLS INTERFACE m_axi port = seed bundle = gmem2 offset = slave

#pragma HLS INTERFACE s_axilite port = underlying bundle = control
#pragma HLS INTERFACE s_axilite port = volatility bundle = control
#pragma HLS INTERFACE s_axilite port = dividendYield bundle = control
#pragma HLS INTERFACE s_axilite port = riskFreeRate bundle = control
#pragma HLS INTERFACE s_axilite port = timeLength bundle = control
#pragma HLS INTERFACE s_axilite port = strike bundle = control
#pragma HLS INTERFACE s_axilite port = optionTypeIn bundle = control
#pragma HLS INTERFACE s_axilite port = resetDates bundle = control
#pragma HLS INTERFACE s_axilite port = seed bundle = control
#pragma HLS INTERFACE s_axilite port = output bundle = control
#pragma HLS INTERFACE s_axilite port = requiredTolerance bundle = control
#pragma HLS INTERFACE s_axilite port = timeSteps bundle = control
#pragma HLS INTERFACE s_axilite port = requiredSamples bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    bool optionType = optionTypeIn;
#ifndef __SYNTHESIS__
    std::cout << "seed[0]=" << seed[0] << std::endl;
    std::cout << "underlying=" << underlying << ",volatility=" << volatility << ",dividendYield=" << dividendYield
              << ",riskFreeRate=" << riskFreeRate << ",timeLength=" << timeLength << ",strike=" << strike
              << ",optionType=" << optionType << ",requiredTolerance=" << requiredTolerance
              << ",requiredSamples=" << requiredSamples << ",timeSteps=" << timeSteps << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "resetDates[" << i << "]=" << resetDates[i] << std::endl;
    }
#endif
    xf::fintech::MCCliquetEngine<TEST_DT, 1>(underlying, volatility, dividendYield, riskFreeRate, timeLength, strike,
                                             optionType, resetDates, seed, output, requiredTolerance, timeSteps,
                                             requiredSamples);
}
