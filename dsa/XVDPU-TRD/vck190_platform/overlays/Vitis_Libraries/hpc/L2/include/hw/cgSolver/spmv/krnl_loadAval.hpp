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
 * @file krnl_loadAval.hpp
 * @brief krnl_loadAval definition.
 *
 * This file is part of Vitis HPC Library.
 */

#ifndef XF_HPC_KRNLLOADAVAL_HPP
#define XF_HPC_KRNLLOADAVAL_HPP

#include "sparse/L2/include/hw/fp64/loadNnz.hpp"
#include "interface.hpp"
#include "krnl_def.hpp"
#include "token.hpp"
/**
 * @brief krnl_loadAval is used to read the values of NNZs out of the device memory
 * @param p_nnzPtr  device memory pointer for reading the values of NNZs
 * @param p_nnzStr output axis stream of NNZ values
 */
extern "C" void krnl_loadAval(CG_interface* p_nnzPtr0,
                              CG_interface* p_nnzPtr1,
                              CG_interface* p_nnzPtr2,
                              CG_interface* p_nnzPtr3,
                              CG_interface* p_nnzPtr4,
                              CG_interface* p_nnzPtr5,
                              CG_interface* p_nnzPtr6,
                              CG_interface* p_nnzPtr7,
                              CG_interface* p_nnzPtr8,
                              CG_interface* p_nnzPtr9,
                              CG_interface* p_nnzPtr10,
                              CG_interface* p_nnzPtr11,
                              CG_interface* p_nnzPtr12,
                              CG_interface* p_nnzPtr13,
                              CG_interface* p_nnzPtr14,
                              CG_interface* p_nnzPtr15,

                              CG_tkStrType& p_tkInStr,
                              CG_wideStrType& p_nnzStr0,
                              CG_wideStrType& p_nnzStr1,
                              CG_wideStrType& p_nnzStr2,
                              CG_wideStrType& p_nnzStr3,
                              CG_wideStrType& p_nnzStr4,
                              CG_wideStrType& p_nnzStr5,
                              CG_wideStrType& p_nnzStr6,
                              CG_wideStrType& p_nnzStr7,
                              CG_wideStrType& p_nnzStr8,
                              CG_wideStrType& p_nnzStr9,
                              CG_wideStrType& p_nnzStr10,
                              CG_wideStrType& p_nnzStr11,
                              CG_wideStrType& p_nnzStr12,
                              CG_wideStrType& p_nnzStr13,
                              CG_wideStrType& p_nnzStr14,
                              CG_wideStrType& p_nnzStr15);
#endif
