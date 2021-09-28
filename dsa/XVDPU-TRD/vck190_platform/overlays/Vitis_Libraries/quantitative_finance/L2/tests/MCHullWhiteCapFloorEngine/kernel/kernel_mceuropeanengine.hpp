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
#ifndef KERNEL_MCEUROPEANENGINHE_H
#define DtUsed double
#define MCM_NM 1
#define OUTDEP 1
#include "xf_fintech/mc_engine.hpp"

extern "C" void kernel_mc_0(unsigned int loop_nm,
                            DtUsed nomial,
                            DtUsed initRate,
                            DtUsed strike,
                            bool isCap,
                            DtUsed singlePeriod,
                            DtUsed alpha,
                            DtUsed sigma,
                            DtUsed output[OUTDEP],
                            DtUsed requiredTolerance,
                            unsigned int requiredSamples,
                            unsigned int timeSteps);
extern "C" void kernel_mc_1(unsigned int loop_nm,
                            DtUsed nomial,
                            DtUsed initRate,
                            DtUsed strike,
                            bool isCap,
                            DtUsed singlePeriod,
                            DtUsed alpha,
                            DtUsed sigma,
                            DtUsed output[OUTDEP],
                            DtUsed requiredTolerance,
                            unsigned int requiredSamples,
                            unsigned int timeSteps);
extern "C" void kernel_mc_2(unsigned int loop_nm,
                            DtUsed nomial,
                            DtUsed initRate,
                            DtUsed strike,
                            bool isCap,
                            DtUsed singlePeriod,
                            DtUsed alpha,
                            DtUsed sigma,
                            DtUsed output[OUTDEP],
                            DtUsed requiredTolerance,
                            unsigned int requiredSamples,
                            unsigned int timeSteps);
extern "C" void kernel_mc_3(unsigned int loop_nm,
                            DtUsed nomial,
                            DtUsed initRate,
                            DtUsed strike,
                            bool isCap,
                            DtUsed singlePeriod,
                            DtUsed alpha,
                            DtUsed sigma,
                            DtUsed output[OUTDEP],
                            DtUsed requiredTolerance,
                            unsigned int requiredSamples,
                            unsigned int timeSteps);
#endif
