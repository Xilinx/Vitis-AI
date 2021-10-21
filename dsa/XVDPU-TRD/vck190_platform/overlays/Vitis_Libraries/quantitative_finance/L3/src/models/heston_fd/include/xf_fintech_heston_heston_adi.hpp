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

#ifndef _XF_FINTECH_HESTON_ADI_H_
#define _XF_FINTECH_HESTON_ADI_H_

#include "heston_types.hpp"

namespace xf {
namespace fintech {
namespace hestonfd {

void xlnx_heston_solve(model_parameters_t* modelParams, two_dimensional_array_t* u);

} // namespace hestonfd
} // namespace fintech
} // namespace xf

#endif //_XF_FINTECH_HESTON_ADI_H_
