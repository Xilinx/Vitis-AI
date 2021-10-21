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

/**
 * @file binomialtreekernel.cpp
 * @brief BinomialTree kernel code.
 */

#include "binomialtree.hpp"

extern "C" {

/**
 * @brief Binomial Tree Kernel entry point
 * @param inputData A vector of model parameters
 * @param outputResult A vector of NPVs
 * @param optionType Calculate for option European/American Call/Put
 * @param numOptions Number of options in the input vector to calculate
 * @param startIndex Can be used with multiple compute units for start index
 * into inputData
 */
void BinomialTreeKernel(xf::fintech::BinomialTreeInputDataType<TEST_DT>* inputData,
                        TEST_DT* outputResult,
                        int optionType,
                        int numOptions,
                        int startIndex) {
#pragma HLS INTERFACE m_axi port = inputData offset = slave bundle = gmem_0
#pragma HLS INTERFACE m_axi port = outputResult offset = slave bundle = gmem_1
#pragma HLS INTERFACE s_axilite port = inputData bundle = control
#pragma HLS INTERFACE s_axilite port = outputResult bundle = control
#pragma HLS INTERFACE s_axilite port = optionType bundle = control
#pragma HLS INTERFACE s_axilite port = numOptions bundle = control
#pragma HLS INTERFACE s_axilite port = startIndex bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS DATA_PACK variable = inputData

    xf::fintech::BinomialTreeInputDataType<TEST_DT> tempInputData[BINOMIAL_TREE_MAX_OPTION_CALCULATIONS];
#pragma HLS DATA_PACK variable = tempInputData
#pragma HLS ARRAY_PARTITION variable = tempInputData cyclic factor = 8 dim = 1

    TEST_DT tempOutputResults[BINOMIAL_TREE_MAX_OPTION_CALCULATIONS];
#pragma HLS ARRAY_PARTITION variable = tempOutputResults cyclic factor = 8 dim = 1

    // Copy all test vectors from global memory to local memory
    for (int i = 0; i < numOptions; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 256
        tempInputData[i] = inputData[i + startIndex];
    }

    // Calculate NPVs
    for (int i = 0; i < numOptions / TEST_PARALLEL_ENGINES; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 32
        for (int j = 0; j < TEST_PARALLEL_ENGINES; j++) {
#pragma HLS UNROLL
            tempOutputResults[i * TEST_PARALLEL_ENGINES + j] =
                xf::fintech::binomialTreeEngine(&tempInputData[i * TEST_PARALLEL_ENGINES + j], optionType);
        }
    }

    // Copy results back to host
    for (int i = 0; i < numOptions; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 256
        outputResult[i + startIndex] = tempOutputResults[i];
    }
}

} // extern "C"
