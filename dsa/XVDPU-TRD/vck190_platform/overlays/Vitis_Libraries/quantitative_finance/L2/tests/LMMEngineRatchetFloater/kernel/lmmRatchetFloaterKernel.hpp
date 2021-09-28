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
 * @brief Header file for wrapper kernel LMM Ratchet floater
 */

#ifndef _LMM_RATCHET_FLOATER_KERNEL_H_
#define _LMM_RATCHET_FLOATER_KERNEL_H_

#include "xf_fintech/lmm_engine.hpp"
#include "xf_fintech/lmm.hpp"

#define TEST_MAX_TENORS (10)
#define TEST_NF (4)
#define TEST_UN (4)
#define TEST_PCA_UN (2)

typedef float TEST_DT;
typedef xf::fintech::lmmRatchetFloaterPricer<TEST_DT, TEST_MAX_TENORS> TEST_PT;

extern "C" void lmmRatchetFloaterKernel(unsigned noTenors,
                                        unsigned noPaths,
                                        TEST_DT* presentRate,
                                        TEST_DT rhoBeta,
                                        TEST_DT* capletVolas,
                                        TEST_DT notional,
                                        TEST_DT rfX,
                                        TEST_DT rfY,
                                        TEST_DT rfAlpha,
                                        ap_uint<32>* seeds,
                                        TEST_DT* outPrice);

#endif // _LMM_RATCHET_FLOATER_KERNEL_H_