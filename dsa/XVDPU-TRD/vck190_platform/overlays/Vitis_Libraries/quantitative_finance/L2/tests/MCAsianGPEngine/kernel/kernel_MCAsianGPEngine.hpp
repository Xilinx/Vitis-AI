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
#ifndef _XF_FINTECH_KERNEL_MCASIANGPENGINE_HPP_
#define _XF_FINTECH_KERNEL_MCASIANGPENGINE_HPP_

typedef double TEST_DT;

extern "C" void kernel_MCAsianGP_0(TEST_DT underlying,
                                   TEST_DT volatility,
                                   TEST_DT dividendYield,
                                   TEST_DT riskFreeRate, // model parameter
                                   TEST_DT timeLength,
                                   TEST_DT strike,
                                   int optionType, // option parameter
                                   TEST_DT outputs[1],
                                   TEST_DT requiredTolerance,
                                   unsigned int requiredSamples,
                                   unsigned int timeSteps,
                                   unsigned int maxSamples);
#endif
