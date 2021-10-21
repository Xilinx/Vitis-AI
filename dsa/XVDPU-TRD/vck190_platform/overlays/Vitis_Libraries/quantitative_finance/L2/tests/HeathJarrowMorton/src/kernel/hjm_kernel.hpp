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

#ifndef _HJM_KERNEL_H_
#define _HJM_KERNEL_H_

#include "xf_fintech/hjm_engine.hpp"
#include "xf_fintech/hjm_model.hpp"

typedef double TEST_DT;

#define TEST_MAX_TENORS (54)
#define TEST_MAX_CURVES (1280)
#define TEST_MC_UN (4)
#define TEST_PCA_NCU (1)

extern "C" void hjm_kernel(TEST_DT* historicalData,
                           unsigned noTenors,
                           unsigned noCurves,
                           float simYears,
                           unsigned noPaths,
                           float zcbMaturity,
                           unsigned* mcSeeds,
                           TEST_DT* outputPrice);

#endif