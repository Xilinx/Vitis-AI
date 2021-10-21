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
#ifndef _MCAE_KERNEL_H_
#define _MCAE_KERNEL_H_

#include "xf_fintech/mc_engine.hpp"
#include "xf_fintech/rng.hpp"

typedef double TEST_DT;
#define TIMESTEPS 100
#define COEF 4
#define UN_K1 2
#define UN_K2_STEP 2
#define UN_K2_PATH 2
#define UN_K3 4
#define iteration 4
#define depthP 1024 * TIMESTEPS* iteration
#define depthM 9 * TIMESTEPS
#define SZ 8 * sizeof(TEST_DT)
#define COEF_DEPTH 1024

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
                        unsigned int timeSteps);

extern "C" void MCAE_k1(TEST_DT timeLength,
                        TEST_DT riskFreeRate,
                        TEST_DT strike,
                        int optionType,
                        ap_uint<8 * sizeof(TEST_DT) * UN_K2_PATH> priceIn[depthP],
                        ap_uint<8 * sizeof(TEST_DT)> matIn[depthM],
                        ap_uint<8 * sizeof(TEST_DT) * COEF> coefOut[COEF_DEPTH],
                        unsigned int calibSamples,
                        unsigned int timeSteps);

extern "C" void MCAE_k2(TEST_DT underlying,
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
                        unsigned int timeSteps);

#endif
