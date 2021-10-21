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

#ifndef _XF_FINTECH_HCF_HOST_H_
#define _XF_FINTECH_HCF_HOST_H_

#define FLOAT
#ifdef FLOAT
#define my_sqrt sqrtf
#define my_atan2 atan2f
#define my_exp expf
#define my_cos cosf
#define my_sin sinf
#define my_log logf
#define my_fabs fabsf
#else
#define my_sqrt sqrt
#define my_atan2 atan2
#define my_exp exp
#define my_cos cos
#define my_sin sin
#define my_log log
#define my_fabs fabs
#endif

#include <vector>
#include "hcf.hpp"
#include "hcf_host.hpp"
#include "xcl2.hpp"

void call_price(std::vector<struct xf::fintech::hcfEngineInputDataType<TEST_DT>,
                            aligned_allocator<struct xf::fintech::hcfEngineInputDataType<TEST_DT> > >& p,
                int num_tests,
                TEST_DT* res);
bool parse_file(std::string file,
                std::vector<struct xf::fintech::hcfEngineInputDataType<TEST_DT>,
                            aligned_allocator<struct xf::fintech::hcfEngineInputDataType<TEST_DT> > >& v,
                TEST_DT dw,
                int w_max,
                int* num_tests,
                TEST_DT* expected_values,
                int max_entries);

#endif
