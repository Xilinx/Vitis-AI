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

extern "C" void MCAE_k0(TEST_DT underlying,
                        TEST_DT volatility,
                        TEST_DT riskFreeRate,
                        TEST_DT dividendYield,
                        TEST_DT timeLength,
                        TEST_DT strike,
                        int optionType,
                        ap_uint<UN_K1 * sizeof(TEST_DT) * 8> pOut[depthP],
                        ap_uint<sizeof(TEST_DT) * 8> mOut[depthM],
                        unsigned int calibSamples,
                        unsigned int timeSteps) {
#pragma HLS INTERFACE m_axi port = pOut bundle = gmem0 offset = slave num_write_outstanding = \
    16 max_write_burst_length = 32
#pragma HLS INTERFACE m_axi port = mOut bundle = gmem1 offset = slave num_write_outstanding = \
    16 max_write_burst_length = 32

#pragma HLS INTERFACE s_axilite port = underlying bundle = control
#pragma HLS INTERFACE s_axilite port = volatility bundle = control
#pragma HLS INTERFACE s_axilite port = riskFreeRate bundle = control
#pragma HLS INTERFACE s_axilite port = dividendYield bundle = control
#pragma HLS INTERFACE s_axilite port = timeLength bundle = control
#pragma HLS INTERFACE s_axilite port = strike bundle = control
#pragma HLS INTERFACE s_axilite port = optionType bundle = control
#pragma HLS INTERFACE s_axilite port = pOut bundle = control
#pragma HLS INTERFACE s_axilite port = mOut bundle = control
#pragma HLS INTERFACE s_axilite port = calibSamples bundle = control
#pragma HLS INTERFACE s_axilite port = timeSteps bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#ifndef __SYNTHESIS__
    std::cout << "underlying is " << underlying << std::endl;
    std::cout << "volatility is " << volatility << std::endl;
    std::cout << "riskFreeRate is " << riskFreeRate << std::endl;
    std::cout << "dividendYield is " << dividendYield << std::endl;
    std::cout << "timeLength is " << timeLength << std::endl;
    std::cout << "strike is " << strike << std::endl;
    std::cout << "optionType is " << optionType << std::endl;
    std::cout << "calibSamples is " << calibSamples << std::endl;
    std::cout << "timeSteps is " << timeSteps << std::endl;
#endif

    bool option = (optionType) ? 1 : 0;
    ap_uint<32> seeds[2];
    seeds[0] = 123456;
    seeds[1] = 567890;

    xf::fintech::MCAmericanEnginePreSamples<TEST_DT, UN_K1>(underlying, volatility, riskFreeRate, dividendYield,
                                                            timeLength, strike, option, seeds, pOut, mOut, calibSamples,
                                                            timeSteps);
}
