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
#include "MCAE_kernel.hpp"
#ifndef __SYNTHESIS__
#include <iostream>
#endif
extern "C" void MCAE_k2(unsigned int seed,
                        TEST_DT underlying,
                        TEST_DT volatility,
                        TEST_DT dividendYield,
                        TEST_DT riskFreeRate,
                        TEST_DT timeLength,
                        TEST_DT strike,
                        int optionType,
                        ap_uint<8 * sizeof(TEST_DT) * COEF> coefIn[COEF_DEPTH],
                        TEST_DT outputs[1],
                        TEST_DT requiredTolerance,
                        unsigned int requiredSamples,
                        unsigned int timeSteps) {
#pragma HLS INTERFACE m_axi port = coefIn bundle = gmem0 offset = slave num_read_outstanding = \
    16 max_read_burst_length = 32
#pragma HLS INTERFACE m_axi port = outputs bundle = gmem1 offset = slave

#pragma HLS INTERFACE s_axilite port = seed bundle = control
#pragma HLS INTERFACE s_axilite port = underlying bundle = control
#pragma HLS INTERFACE s_axilite port = volatility bundle = control
#pragma HLS INTERFACE s_axilite port = dividendYield bundle = control
#pragma HLS INTERFACE s_axilite port = riskFreeRate bundle = control
#pragma HLS INTERFACE s_axilite port = timeLength bundle = control
#pragma HLS INTERFACE s_axilite port = strike bundle = control
#pragma HLS INTERFACE s_axilite port = optionType bundle = control
#pragma HLS INTERFACE s_axilite port = coefIn bundle = control
#pragma HLS INTERFACE s_axilite port = outputs bundle = control
#pragma HLS INTERFACE s_axilite port = requiredTolerance bundle = control
#pragma HLS INTERFACE s_axilite port = requiredSamples bundle = control
#pragma HLS INTERFACE s_axilite port = timeSteps bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#ifndef __SYNTHESIS__
    std::cout << "underlying is " << underlying << std::endl;
    std::cout << "volatility is " << volatility << std::endl;
    std::cout << "dividendYield is " << dividendYield << std::endl;
    std::cout << "riskFreeRate is " << riskFreeRate << std::endl;
    std::cout << "timeLength is " << timeLength << std::endl;
    std::cout << "strike is " << strike << std::endl;
    std::cout << "optionType is " << optionType << std::endl;
    std::cout << "requiredTolerance is " << requiredTolerance << std::endl;
    std::cout << "requiredSamples is " << requiredSamples << std::endl;
    std::cout << "timeSteps is " << timeSteps << std::endl;
#endif

    bool option = (optionType) ? 1 : 0;

    ap_uint<32> seeds[4];
    seeds[0] = seed;
    seeds[1] = seed * 3;
    seeds[2] = seed * 6;
    seeds[3] = seed * 9;

    xf::fintech::MCAmericanEnginePricing<double, UN_K3>(underlying, volatility, dividendYield, riskFreeRate, timeLength,
                                                        strike, option, seeds, coefIn, outputs, requiredTolerance,
                                                        requiredSamples, timeSteps);
}
