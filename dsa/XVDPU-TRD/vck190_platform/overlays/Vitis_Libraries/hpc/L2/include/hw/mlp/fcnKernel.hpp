/**********
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
 * **********/

#ifndef XF_HPC_MLP_FCNKERNEL_HPP
#define XF_HPC_MLP_FCNKERNEL_HPP

#include "ddr.hpp"
#include "ddrType.hpp"
#include "fcn.hpp"

// Compute engine types

#if BLAS_runFcn == 1
typedef xf::hpc::mlp::Fcn<BLAS_dataType,
                          BLAS_XdataType,
                          BLAS_ddrWidth,
                          BLAS_XddrWidth,
                          BLAS_gemmKBlocks,
                          BLAS_gemmMBlocks,
                          BLAS_gemmNBlocks
#if BLAS_CACHE == 1
                          ,
                          512 * 32,
                          32
#endif
                          >
    FcnType;
#endif

/**
 * @brief fcnKernel defines the kernel top function, with DDR/HBM as an interface
 *
 * @param p_DdrRd is DDR/HBM memory address used for read
 * @param p_DdrWr is DDR/HBM memory address used for write
 */
extern "C" {
void fcnKernel(DdrIntType* p_DdrRd, DdrIntType* p_DdrWr);
} // extern C

#endif
