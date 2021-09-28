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
#pragma once

#include "xf_blas.hpp"
#include "rtm.hpp"
#include "params.hpp"

using namespace xf::hpc;
using namespace rtm;
typedef RTM2D<DATATYPE, ORDER, 1290, NXB, nPE> RTM_TYPE;
typedef RTM_TYPE::t_WideType WIDE_TYPE;
typedef RTM_TYPE::t_PairType PAIR_TYPE;
typedef RTM_TYPE::t_PairInType PAIRIN_TYPE;
typedef RTM_TYPE::t_InType IN_TYPE;
typedef typename RTM_TYPE::t_UpbInType UPB_TYPE;

extern "C" void top(const bool,
                    const unsigned int p_z,
                    const unsigned int p_x,
                    const unsigned int p_t,
                    const unsigned int p_srcz,
                    const unsigned int p_srcx,
                    const DATATYPE* p_src,
                    const DATATYPE* p_coefz,
                    const DATATYPE* p_coefx,
                    const DATATYPE* p_taperz,
                    const DATATYPE* p_taperx,
                    const IN_TYPE* p_v2dt2,
                    PAIRIN_TYPE* p_p0,
                    PAIRIN_TYPE* p_p1,
                    PAIRIN_TYPE* p_p2,
                    PAIRIN_TYPE* p_p3,
                    UPB_TYPE* p_upb);
