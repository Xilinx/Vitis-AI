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

/**
 * @file krnl_loadArbParam.hpp
 * @brief krnl_loadArbParam definition.
 *
 * This file is part of Vitis HPC Library.
 */

#ifndef XF_HPC_KRNLLOADARBPARAM_HPP
#define XF_HPC_KRNLLOADARBPARAM_HPP

#include "sparse/L1/include/hw/xf_sparse_fp64.hpp"
#include "interface.hpp"
#include "krnl_def.hpp"
#include "token.hpp"

/**
 * @brief krnl_loadArbParam is used to read the row block parameters out of the device memory
 * @param p_rbParamPtr  device memory pointer for reading the row block params
 * @param p_chRbParamStr output axis streams of channel row block parameters
 * @param p_rbParamStr output axis stream of row block parameters
 */
extern "C" void krnl_loadArbParam(CG_interface* p_rbParamPtr,
                                  CG_tkStrType& p_tkInStr,
                                  CG_wideParamStrType& p_chRbParamStr,
                                  CG_paramStrType& p_rbParamStr);

#endif
