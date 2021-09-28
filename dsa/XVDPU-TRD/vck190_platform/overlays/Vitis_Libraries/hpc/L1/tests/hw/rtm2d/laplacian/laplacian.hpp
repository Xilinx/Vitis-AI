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
#include "xf_blas.hpp"
#include "rtm.hpp"
#include "params.hpp"

typedef xf::hpc::rtm::Stencil2D<DATATYPE, ORDER, MaxD, nPE> STENCEIL_TYPE;
typedef STENCEIL_TYPE::t_InType t_InType;
typedef STENCEIL_TYPE::t_WideType t_WideType;

extern "C" void top(const unsigned int p_z,
                    const unsigned int p_x,
                    const DATATYPE* p_coefz,
                    const DATATYPE* p_coefx,
                    const t_InType* p_img,
                    t_InType* p_out);
