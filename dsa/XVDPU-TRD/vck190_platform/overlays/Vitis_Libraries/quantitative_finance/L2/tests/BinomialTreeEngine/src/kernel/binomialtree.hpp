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

#ifndef _XF_FINTECH_BINOMIALTREE_HPP_
#define _XF_FINTECH_BINOMIALTREE_HPP_
// include the binomial engine
#include "xf_fintech/bt_engine.hpp"

// This should match the number of CU within the kernel
#define BINOMIAL_TREE_CU_PER_KERNEL (1)

// Kernel name will change based on configuration
#define BINOMIAL_TREE_KERNEL_NAME "BinomialTreeKernel"
#define BINOMIAL_TREE_MAX_OPTION_CALCULATIONS (1024)

// CPU compare function
extern void BinomialTreeCPU(int optionType,
                            xf::fintech::BinomialTreeInputDataType<TEST_DT>* inputData,
                            TEST_DT* outputResult,
                            int numOptions);

#endif
