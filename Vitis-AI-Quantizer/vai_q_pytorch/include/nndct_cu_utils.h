

/* 
# (c) Copyright 2016 – 2019 Xilinx, Inc. All rights reserved. 
# 
# This file contains confidential and proprietary information 
# of Xilinx, Inc. and is protected under U.S. and 
# international copyright and other intellectual property
# laws.
# 
# DISCLAIMER
# This disclaimer is not a license and does not grant any
# rights to the materials distributed herewith. Except as
# otherwise provided in a valid license issued to you by
# Xilinx, and to the maximum extent permitted by applicable
# law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
# WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
# AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
# BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
# INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE;
# (2) Xilinx shall not be liable (whether in contract or tort,
# including negligence, or under any other theory of
# liability) for any loss or damage of any kind or nature
# related to, arising under or in connection with these
# materials, including for any direct, or any indirect,
# special, incidental, or consequential loss or damage
# (including loss of data, profits, goodwill, or any type of
# loss or damage suffered as a result of any action brought
# by a third party) even if such damage or loss was
# reasonably foreseeable or Xilinx had been advised of the
# possibility of the same.
# 
# CRITICAL APPLICATIONS
# Xilinx products are not designed or intended to be fail-
# safe, or for use in any application requiring fail-safe
# performance, such as life-support or safety devices or
# systems, Class III medical devices, nuclear facilities,
# applications related to the deployment of airbags, or any
# other applications that could lead to death, personal
# injury, or severe property or environmental damage
# (individually and collectively, "Critical
# Applications"). Customer assumes the sole risk and
# liability of any use of Xilinx products in Critical
# Applications, subject only to applicable laws and
# regulations governing limitations on product liability.
# 
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
# PART OF THIS FILE AT ALL TIMES
*/


#ifndef _NNDCT_CU_UTILS_H_
#define _NNDCT_CU_UTILS_H_
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <algorithm>
#include <assert.h>

#define BLOCKSIZE_COL 64
#define BLOCKSIZE_ROW 4
#define BLOCKSIZE 256
#define NNDCT_CUDA_NUM_THREADS 512
#define CU2DBLOCK 16
#define CU1DBLOCK 256

#define NNDCT_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)



inline int NNDCT_GET_BLOCKS(const int N){
  return (N + NNDCT_CUDA_NUM_THREADS - 1) / NNDCT_CUDA_NUM_THREADS;
}

inline int n_blocks(int size, int block_size) {
  return size / block_size + ((size % block_size == 0)? 0 : 1);
}

inline int NNDCT_GET_BLOCKS1D(const int N){
  int dimGrid = n_blocks(N, CU1DBLOCK);
  if (dimGrid > 256) {
    dimGrid = 256;
  }
  return dimGrid;
}

dim3 GetGridSizeF(unsigned n);

void GetBlockSizesForSimpleMatrixOperation(int num_rows,
    int num_cols,dim3 *dimGrid,dim3 *dimBlock);




#endif //_NNDCT_CU_UTILS_H_
