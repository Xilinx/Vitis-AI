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
#ifndef _XF_FINTECH_MCENGINE_TOP_HPP_
#define _XF_FINTECH_MCENGINE_TOP_HPP_

#include "xf_fintech/enums.hpp"
#include "xf_fintech/mc_engine.hpp"
#include "xf_fintech/rng.hpp"
typedef float TEST_DT;

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
                          unsigned int timeSteps);
#endif
