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

#ifndef _HWA_ENGINE_KERNEL_H_
#define _HWA_ENGINE_KERNEL_H_

#include "xf_fintech/hwa_engine.hpp"

using namespace xf::fintech;

#define N_k0 (16)
#define N_k1 (16)
#define N_k2 (16)
#define LEN (12) // length of yield curve data

extern "C" void HWA_k0(TEST_DT a,
                       TEST_DT sigma,
                       TEST_DT times[LEN],
                       TEST_DT rates[LEN],
                       TEST_DT t[N_k0],
                       TEST_DT T[N_k0],
                       TEST_DT P[N_k0]);

extern "C" void HWA_k1(TEST_DT a,
                       TEST_DT sigma,
                       TEST_DT times[LEN],
                       TEST_DT rates[LEN],
                       int types[N_k1],
                       TEST_DT t[N_k1],
                       TEST_DT T[N_k1],
                       TEST_DT S[N_k1],
                       TEST_DT K[N_k1],
                       TEST_DT P[N_k1]);

extern "C" void HWA_k2(TEST_DT a,
                       TEST_DT sigma,
                       TEST_DT times[LEN],
                       TEST_DT rates[LEN],
                       int capfloorType[N_k2],
                       TEST_DT startYear[N_k2],
                       TEST_DT endYear[N_k2],
                       int settlementFreq[N_k2],
                       TEST_DT N[N_k2],
                       TEST_DT X[N_k2],
                       TEST_DT P[N_k2]);

#endif /* _HWA_ENGINE_KERNEL_H_ */
