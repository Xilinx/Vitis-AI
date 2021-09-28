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

#include "ap_fixed.h"
#include "xf_fintech/enums.hpp"
#include "xf_fintech/mc_engine.hpp"
#include "xf_fintech/rng.hpp"

typedef double TEST_DT;

extern "C" void MCEHGEngine_k0(TEST_DT underlying,
                               TEST_DT riskFreeRate,
                               TEST_DT sigma,
                               TEST_DT v0,
                               TEST_DT theta,
                               TEST_DT kappa,
                               TEST_DT rho,
                               TEST_DT dividendYield,
                               unsigned int optionType,
                               TEST_DT strike,
                               TEST_DT timeLength,
                               unsigned int timeSteps,
                               unsigned int requiredSamples,
                               unsigned int maxSamples,
                               ap_uint<32> seed[8 * 2],
                               TEST_DT requiredTolerance,
                               TEST_DT outputs[8]);

#endif
