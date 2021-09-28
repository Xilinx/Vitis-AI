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
 * @file krnl_loadIdx.hpp
 * @brief krnl_loadIdx definition.
 *
 * This file is part of Vitis HPC Library.
 */

#ifndef XF_HPC_KRNLLOADIDX_HPP
#define XF_HPC_KRNLLOADIDX_HPP

#include "sparse/L2/include/hw/fp64/loadIdx.hpp"
#include "interface.hpp"
#include "krnl_def.hpp"
#include "token.hpp"

/**
 * @brief krnl_loadIdx is used to read the indices of NNZs out of the device memory
 * @param p_idxPtr  device memory pointer for reading the indices of NNZs
 * @param p_idxStr output axis stream of NNZ indices
 */
extern "C" void krnl_loadAidx(CG_interface* p_idxPtr,
                              CG_tkStrType& p_tkInStr,
                              CG_idxStrType& p_idxStr0,
                              CG_idxStrType& p_idxStr1,
                              CG_idxStrType& p_idxStr2,
                              CG_idxStrType& p_idxStr3,
                              CG_idxStrType& p_idxStr4,
                              CG_idxStrType& p_idxStr5,
                              CG_idxStrType& p_idxStr6,
                              CG_idxStrType& p_idxStr7,
                              CG_idxStrType& p_idxStr8,
                              CG_idxStrType& p_idxStr9,
                              CG_idxStrType& p_idxStr10,
                              CG_idxStrType& p_idxStr11,
                              CG_idxStrType& p_idxStr12,
                              CG_idxStrType& p_idxStr13,
                              CG_idxStrType& p_idxStr14,
                              CG_idxStrType& p_idxStr15);
#endif
