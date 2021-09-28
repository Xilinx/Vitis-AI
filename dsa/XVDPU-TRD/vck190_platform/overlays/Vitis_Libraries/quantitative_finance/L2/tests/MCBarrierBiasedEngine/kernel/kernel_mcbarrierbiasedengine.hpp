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
#ifndef _KERNEL_MCBARRIERBIASENGINE_H_
//#include <ap_int.h>
#define DtUsed float
#define OUTDEP 1024
#define MCM_NM 1

extern "C" void McBarrierBiasedEngine_k(unsigned int loopNum,
                                        DtUsed underlying,
                                        DtUsed volatility,
                                        DtUsed dividendYield,
                                        DtUsed riskFreeRate,
                                        DtUsed timeLength, // model parameter
                                        DtUsed barrier,
                                        DtUsed strike,
                                        // ap_uint<2> barrierType,
                                        int optionType, // option parameter
                                        DtUsed out[OUTDEP],
                                        DtUsed rebate,
                                        DtUsed requiredTolerance,
                                        unsigned int requiredSamples,
                                        unsigned int timeSteps);

#endif
