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

#include "xf_fintech_heston_solver_parameters.hpp"

namespace xf {
namespace fintech {

HestonFDSolverParameters::HestonFDSolverParameters(HestonFDModelParameters& ModelParameters) {
    _Theta = _Default_Theta;
    _N = _Default_N;
    _m1 = _Default_m1;
    _m2 = _Default_m2;
    _dt = ModelParameters.Get_T() / _N;
}

} // namespace fintech
} // namespace xf
