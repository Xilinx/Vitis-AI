/*
 *
 * * Copyright 2019 Xilinx Inc.
 * *
 * * Licensed under the Apache License, Version 2.0 (the "License");
 * * you may not use this file except in compliance with the License.
 * * You may obtain a copy of the License at
 * *
 * *     http://www.apache.org/licenses/LICENSE-2.0
 * *
 * * Unless required by applicable law or agreed to in writing, software
 * * distributed under the License is distributed on an "AS IS" BASIS,
 * * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * * See the License for the specific language governing permissions and
 * * limitations under the License.
 * */

#ifndef _NNDCT_CU_UTILS_H_
#define _NNDCT_CU_UTILS_H_
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <algorithm>
#include <assert.h>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#include <cuda_runtime.h>
#endif

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