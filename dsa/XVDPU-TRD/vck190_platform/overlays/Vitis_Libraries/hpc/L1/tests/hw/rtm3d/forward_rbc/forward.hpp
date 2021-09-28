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

typedef Domain3D<MaxZ, MaxY, ORDER / 2, nPE, nPE, NUM_INST> DOMAIN_TYPE;
typedef RTM3D<DOMAIN_TYPE, DATATYPE, ORDER, MaxZ, MaxY, MaxB, nPE, nPE> RTM_TYPE;
typedef RTM_TYPE::t_DataTypeX TYPEX;
typedef RTM_TYPE::t_WideType WIDE_TYPE;
typedef RTM_TYPE::t_InType IN_TYPE;

extern "C" void top(const bool p_sel,
                    const unsigned int p_z,
                    const unsigned int p_y,
                    const unsigned int p_x,
                    const unsigned int p_t,
                    const unsigned int p_srcz,
                    const unsigned int p_srcy,
                    const unsigned int p_srcx,
                    const DATATYPE* p_src,
                    const DATATYPE* p_coefz,
                    const DATATYPE* p_coefy,
                    const DATATYPE* p_coefx,
                    const IN_TYPE* p_v2dt2,
                    IN_TYPE* p_pi0,
                    IN_TYPE* p_pi1,
                    IN_TYPE* p_po0,
                    IN_TYPE* p_po1,
                    IN_TYPE* p_ppi0,
                    IN_TYPE* p_ppi1,
                    IN_TYPE* p_ppo0,
                    IN_TYPE* p_ppo1);
