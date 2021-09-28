
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
 * @file krnl_loadPkApar.hpp
 * @brief krnl_loadPkApar definition.
 *
 * This file is part of Vitis HPC Library.
 */

#ifndef XF_SPARSE_KRNLLOADPKAPAR_HPP
#define XF_SPARSE_KRNLLOADPKAPAR_HPP

#include "sparse/L1/include/hw/xf_sparse_fp64.hpp"
#include "interface.hpp"
#include "krnl_def.hpp"
#include "token.hpp"

/**
 * @brief krnl_loadPkApar is used to read the input vector x out of device memory
 * @param p_parParamPtr  device memory pointer for reading the partition parameters
 * @param p_xPtr device memory pointer for reading vector x
 * @param p_paramStr output axis stream of partition parameters
 * @param p_xStr    output axis stream of x entries
 */
extern "C" void krnl_loadPkApar(CG_interface* p_parParamPtr,
                                CG_interface* p_xPtr,
                                CG_tkStrType& p_tkInStr,
                                CG_wideParamStrType& p_paramStr,
                                CG_wideStrType& p_outXstr);
#endif
