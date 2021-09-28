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

extern "C" void MCAE_k1(TEST_DT timeLength,
                        TEST_DT riskFreeRate,
                        TEST_DT strike,
                        int optionType,
                        ap_uint<8 * sizeof(TEST_DT) * UN_K2_PATH> priceIn[depthP],
                        ap_uint<8 * sizeof(TEST_DT)> matIn[depthM],
                        ap_uint<8 * sizeof(TEST_DT) * COEF> coefOut[COEF_DEPTH],
                        unsigned int calibSamples,
                        unsigned int timeSteps) {
#pragma HLS INTERFACE m_axi port = priceIn bundle = gmem0 offset = slave num_read_outstanding = \
    16 max_read_burst_length = 32
#pragma HLS INTERFACE m_axi port = matIn bundle = gmem1 offset = slave num_read_outstanding = \
    16 max_read_burst_length = 32
#pragma HLS INTERFACE m_axi port = coefOut bundle = gmem2 offset = slave num_write_outstanding = \
    1 max_write_burst_length = 8

#pragma HLS INTERFACE s_axilite port = timeLength bundle = control
#pragma HLS INTERFACE s_axilite port = riskFreeRate bundle = control
#pragma HLS INTERFACE s_axilite port = strike bundle = control
#pragma HLS INTERFACE s_axilite port = optionType bundle = control
#pragma HLS INTERFACE s_axilite port = priceIn bundle = control
#pragma HLS INTERFACE s_axilite port = matIn bundle = control
#pragma HLS INTERFACE s_axilite port = coefOut bundle = control
#pragma HLS INTERFACE s_axilite port = calibSamples bundle = control
#pragma HLS INTERFACE s_axilite port = timeSteps bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#ifndef __SYNTHESIS__
    std::cout << "timeLength is " << timeLength << std::endl;
    std::cout << "riskFreeRate is " << riskFreeRate << std::endl;
    std::cout << "strike is " << strike << std::endl;
    std::cout << "optionType is " << optionType << std::endl;
    std::cout << "calibSamples is " << calibSamples << std::endl;
    std::cout << "timeSteps is " << timeSteps << std::endl;
#endif

    bool option = (optionType) ? 1 : 0;

    xf::fintech::MCAmericanEngineCalibrate<TEST_DT, UN_K2_PATH, UN_K2_STEP>(
        timeLength, riskFreeRate, strike, option, priceIn, matIn, coefOut, calibSamples, timeSteps);
}
