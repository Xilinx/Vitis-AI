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
 * @file binomialtreecpu.cpp
 * @brief CPU version of engine.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <thread>

#include "binomialtree.hpp"

using namespace std;
using namespace xf::fintech;
using namespace internal;

// test data
#include "../../data/bt_testcases.hpp"

/**
 * @brief Binomial CPU Processing Element
 * @param optionType American/European Call or put
 * @param inputData Structure containing the input data model parameters
 * @return A single NPV value
 */
TEST_DT BinomialTreeCPUEngine(int optionType, BinomialTreeInputDataType<TEST_DT>* inputData) {
    const int max_supported_nodes = 1024;

    TEST_DT options[max_supported_nodes];
    TEST_DT deltaT = TEST_DT(inputData->T / inputData->N);
    TEST_DT upFactor = EXP(inputData->V * SQRT(deltaT));
    TEST_DT pUp =
        (upFactor * EXP(-inputData->q * deltaT) - EXP(-inputData->rf * deltaT)) / (POW(upFactor, TEST_DT(2)) - 1);
    TEST_DT pDown = EXP(-inputData->rf * deltaT) - pUp;

    // Fair price for option calcaulated at the final step
    for (int i = 0; i <= inputData->N; i++) {
        if (optionType == BinomialTreeAmericanPut || optionType == BinomialTreeEuropeanPut) {
            options[i] = inputData->K - (inputData->S * POW(upFactor, TEST_DT(2 * i - inputData->N)));
        } else {
            options[i] = (inputData->S * POW(upFactor, TEST_DT(2 * i - inputData->N))) - inputData->K;
        }

        options[i] = MAX(options[i], TEST_DT(0.0));
    }

    // Work back from the leaf nodes to get the NPV
    for (int j = (inputData->N - 1); j >= 0; j--) {
        for (int i = 0; i <= j; i++) {
            // Binomial Value - represents the fair price of the derivative at a
            // particular point in time
            options[i] = TEST_DT((pUp * options[i + 1]) + (pDown * options[i]));

            if (optionType == BinomialTreeAmericanCall || optionType == BinomialTreeAmericanPut) {
                TEST_DT exercise;

                if (optionType == BinomialTreeAmericanPut) {
                    exercise = inputData->K - inputData->S * POW(upFactor, TEST_DT(2 * i - j));
                } else if (optionType == BinomialTreeAmericanCall) {
                    exercise = inputData->S * POW(upFactor, TEST_DT(2 * i - j)) - inputData->K;
                }

                options[i] = MAX(options[i], exercise);
            }
        }
    }

    // Return calculated NPV
    return (options[0]);
}

/**
 * @brief Binomial CPU entry point
 * @param optionType American or European Call/Put
 * @param inputData Vector of model parameters
 * @param outputResult Vector of NPV results
 * @param numOptions Number of NPV to calculate
 */
void BinomialTreeCPU(int optionType,
                     BinomialTreeInputDataType<TEST_DT>* inputData,
                     TEST_DT* outputResult,
                     int numOptions) {
    BinomialTreeInputDataType<TEST_DT> tempInputData[BINOMIAL_TREE_MAX_OPTION_CALCULATIONS];
    TEST_DT tempOutputResults[BINOMIAL_TREE_MAX_OPTION_CALCULATIONS];

    // follow same approach used by kernel
    for (int i = 0; i < numOptions; i++) {
        tempInputData[i] = inputData[i];
    }

    // Calculate NPVs
    for (int i = 0; i < numOptions; i++) {
        tempOutputResults[i] = BinomialTreeCPUEngine(optionType, &tempInputData[i]);
    }

    for (int i = 0; i < numOptions; i++) {
        outputResult[i] = tempOutputResults[i];
    }
}
