
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
 * @file loadColKernel.hpp
 * @brief loadColKernel definition.
 *
 * This file is part of Vitis SPARSE Library.
 */

#ifndef XF_SPARSE_LOADPARXKERNEL_HPP
#define XF_SPARSE_LOADPARXKERNEL_HPP

#include "kernel.hpp"

/**
 * @brief loadParXkernel is used to read the input vector X and partition parameters out of device memory
 * @param p_parParamPtr  device memory pointer for reading the partition parameters
 * @param p_xPtr device memory pointer for reading vector X
 * @param p_paramStr output axis stream of partition parameters
 * @param p_xStr    output axis stream of X entries
 */
extern "C" void loadParXkernel(HBM_InfTyp* p_parParamPtr,
                               HBM_InfTyp* p_xPtr,
                               WideParamStrTyp& p_paramStr,
                               HBM_StrTyp& p_outXstr);
#endif
