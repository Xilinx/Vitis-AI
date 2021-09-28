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

extern "C" void kernel_mc_0(unsigned int loop_nm,
                            DtUsed underlying,
                            DtUsed riskFreeRate,
                            DtUsed sigma,
                            DtUsed v0,
                            DtUsed theta,
                            DtUsed kappa,
                            DtUsed rho,
                            DtUsed dividendYield,
                            bool optionType,
                            DtUsed strike,
                            DtUsed timeLength,
                            unsigned int timeSteps,
                            unsigned int requiredSamples,
                            unsigned int maxSamples,
                            DtUsed requiredTolerance,
                            DtUsed outputs[OUTDEP]);
extern "C" void kernel_mc_1(unsigned int loop_nm,
                            DtUsed underlying,
                            DtUsed riskFreeRate,
                            DtUsed sigma,
                            DtUsed v0,
                            DtUsed theta,
                            DtUsed kappa,
                            DtUsed rho,
                            DtUsed dividendYield,
                            bool optionType,
                            DtUsed strike,
                            DtUsed timeLength,
                            unsigned int timeSteps,
                            unsigned int requiredSamples,
                            unsigned int maxSamples,
                            DtUsed requiredTolerance,
                            DtUsed outputs[OUTDEP]);
extern "C" void kernel_mc_2(unsigned int loop_nm,
                            DtUsed underlying,
                            DtUsed riskFreeRate,
                            DtUsed sigma,
                            DtUsed v0,
                            DtUsed theta,
                            DtUsed kappa,
                            DtUsed rho,
                            DtUsed dividendYield,
                            bool optionType,
                            DtUsed strike,
                            DtUsed timeLength,
                            unsigned int timeSteps,
                            unsigned int requiredSamples,
                            unsigned int maxSamples,
                            DtUsed requiredTolerance,
                            DtUsed outputs[OUTDEP]);
extern "C" void kernel_mc_3(unsigned int loop_nm,
                            DtUsed underlying,
                            DtUsed riskFreeRate,
                            DtUsed sigma,
                            DtUsed v0,
                            DtUsed theta,
                            DtUsed kappa,
                            DtUsed rho,
                            DtUsed dividendYield,
                            bool optionType,
                            DtUsed strike,
                            DtUsed timeLength,
                            unsigned int timeSteps,
                            unsigned int requiredSamples,
                            unsigned int maxSamples,
                            DtUsed requiredTolerance,
                            DtUsed outputs[OUTDEP]);
#endif
