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
#define KERNEL_MCEUROPEANENGINHE_H

#define DtUsed double
#define MCM_NM 8
#define OUTDEP 1024

extern "C" void kernel_mc(unsigned int loop_nm,
                          unsigned int seed,
                          DtUsed underlying,
                          DtUsed volatility,
                          DtUsed dividendYield,
                          DtUsed riskFreeRate, // model parameter
                          DtUsed timeLength,
                          DtUsed strike,
                          unsigned int optionType, // option parameter
                          DtUsed out[OUTDEP],
                          DtUsed requiredTolerance,
                          unsigned int requiredSamples,
                          unsigned int timeSteps,
                          unsigned int maxSamples);
#endif
