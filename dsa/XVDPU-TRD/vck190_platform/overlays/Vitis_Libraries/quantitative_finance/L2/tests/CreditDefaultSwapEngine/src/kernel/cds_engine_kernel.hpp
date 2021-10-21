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

#ifndef _CDS_ENGINE_KERNEL_H_
#define _CDS_ENGINE_KERNEL_H_

#include "xf_fintech/cds_engine.hpp"

using namespace xf::fintech;

#define IRLEN (21)
#define HAZARDLEN (6)
#define N (8)

extern "C" void CDS_kernel(TEST_DT timesIR[IRLEN],
                           TEST_DT ratesIR[IRLEN],
                           TEST_DT timesHazard[HAZARDLEN],
                           TEST_DT ratesHazard[HAZARDLEN],
                           TEST_DT notional[N],
                           TEST_DT recovery[N],
                           TEST_DT maturity[N],
                           int frequency[N],
                           TEST_DT cdsSpread[N]);

#endif /* _CDS_ENGINE_KERNEL_H_ */
