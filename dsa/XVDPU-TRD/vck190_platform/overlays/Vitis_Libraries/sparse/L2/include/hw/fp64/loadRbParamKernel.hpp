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
 * @file loadRbParamKernel.hpp
 * @brief loadRbParamKernel definition.
 *
 * This file is part of Vitis SPARSE Library.
 */

#ifndef XF_SPARSE_LOADRBPARAMKERNEL_HPP
#define XF_SPARSE_LOADRBPARAMKERNEL_HPP

#include "kernel.hpp"

/**
 * @brief loadRbParamKernel is used to read the row block parameters out of the device memory
 * @param p_rbParamPtr  device memory pointer for reading the row block parameters
 * @param p_chRbParamStr output axis streams of channel row block parameters
 * @param p_rbParamStr output axis stream of row block parameters
 */
extern "C" void loadRbParamKernel(HBM_InfTyp* p_rbParamPtr, WideParamStrTyp& p_chRbParamStr, ParamStrTyp& p_rbParamStr);

#endif
