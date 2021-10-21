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
 * @file loadNnzKernel.hpp
 * @brief loadNnzKernel definition.
 *
 * This file is part of Vitis SPARSE Library.
 */

#ifndef XF_SPARSE_LOADNNZKERNEL_HPP
#define XF_SPARSE_LOADNNZKERNEL_HPP

#include "kernel.hpp"
#include "loadNnz.hpp"

/**
 * @brief loadNnzKernel is used to read the values of NNZs out of the device memory
 * @param p_nnzPtr  device memory pointer for reading the values of NNZs
 * @param p_nnzStr output axis stream of NNZ values
 */
extern "C" void loadNnzKernel(HBM_InfTyp* p_nnzPtr0,
                              HBM_InfTyp* p_nnzPtr1,
                              HBM_InfTyp* p_nnzPtr2,
                              HBM_InfTyp* p_nnzPtr3,
                              HBM_InfTyp* p_nnzPtr4,
                              HBM_InfTyp* p_nnzPtr5,
                              HBM_InfTyp* p_nnzPtr6,
                              HBM_InfTyp* p_nnzPtr7,
                              HBM_InfTyp* p_nnzPtr8,
                              HBM_InfTyp* p_nnzPtr9,
                              HBM_InfTyp* p_nnzPtr10,
                              HBM_InfTyp* p_nnzPtr11,
                              HBM_InfTyp* p_nnzPtr12,
                              HBM_InfTyp* p_nnzPtr13,
                              HBM_InfTyp* p_nnzPtr14,
                              HBM_InfTyp* p_nnzPtr15,

                              HBM_StrTyp& p_nnzStr0,
                              HBM_StrTyp& p_nnzStr1,
                              HBM_StrTyp& p_nnzStr2,
                              HBM_StrTyp& p_nnzStr3,
                              HBM_StrTyp& p_nnzStr4,
                              HBM_StrTyp& p_nnzStr5,
                              HBM_StrTyp& p_nnzStr6,
                              HBM_StrTyp& p_nnzStr7,
                              HBM_StrTyp& p_nnzStr8,
                              HBM_StrTyp& p_nnzStr9,
                              HBM_StrTyp& p_nnzStr10,
                              HBM_StrTyp& p_nnzStr11,
                              HBM_StrTyp& p_nnzStr12,
                              HBM_StrTyp& p_nnzStr13,
                              HBM_StrTyp& p_nnzStr14,
                              HBM_StrTyp& p_nnzStr15);

#endif
