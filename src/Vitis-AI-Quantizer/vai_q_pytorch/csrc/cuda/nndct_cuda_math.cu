

/*
* Copyright 2019 Xilinx Inc.
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


#include <math.h>
#include <algorithm>
#include "../../include/cuda/nndct_fix_kernels.cuh"
#include "../../include/cuda/nndct_cu_utils.h"
#include "../../include/cuda/nndct_cuda_math.h"

#ifdef __HIP_PLATFORM_AMD__
#define CUDART_INF_F            __int_as_float(0x7f800000)
#define CUDART_INF              __longlong_as_double(0x7ff0000000000000ULL)
#else
#include <math_constants.h>
#endif

template<typename Dtype>
__global__ static void _set(const int N, 
                            Dtype* data, 
                            Dtype val){
  NNDCT_KERNEL_LOOP(index, N){
    data[index] = val;
  }
}

template<typename Dtype>
__global__ static void _scale_inplace(const int N, 
                                      Dtype* data, 
                                      Dtype scale){
  NNDCT_KERNEL_LOOP(index, N){
    data[index] *= scale;
  }
}

template<typename Dtype>
__global__ static void _scale(const int N,
                              const Dtype* src,
                              Dtype* dst,
                              Dtype scale){
  NNDCT_KERNEL_LOOP(index, N){
    dst[index] = scale * src[index];
  }
}

template<typename Dtype>
__global__ static void _sub(const int N, 
                            const Dtype* src, 
                            Dtype* dst){
  NNDCT_KERNEL_LOOP(index, N){
    dst[index] = src[index] - dst[index];
  }
}

template<typename Dtype>
__global__ static void _pow(const int N,
                            Dtype* data,
                            Dtype power){
  NNDCT_KERNEL_LOOP(index, N){
    data[index] = pow(data[index], power);
  }
}

//from kaldi, reduction without device handle
enum EnumTransformReduce {
  SUMAB, SUM, MAX, MIN, LINFNORM, L2NORM, L1NORM, L0NORM, LPNORM
};

template<EnumTransformReduce TransReduceType, typename Dtype>
struct TransReduceOp {
  __forceinline__
  __device__ Dtype InitValue() const {
    return Dtype(0);
  }
  __forceinline__
  __device__ Dtype Transform(const Dtype& x) const {
    return Dtype(0);
  }
  __forceinline__
  __device__ Dtype Reduce(const Dtype& a, const Dtype& b) const {
    return Dtype(0);
  }
  __forceinline__
  __device__ Dtype PostReduce(const Dtype& x, const Dtype& output) const {
    return Dtype(0);
  }
};

template<typename Dtype>
struct TransReduceOp<SUM, Dtype> {
  __forceinline__
  __device__ Dtype InitValue() const {
    return Dtype(0);
  }
  __forceinline__
  __device__ Dtype Transform(const Dtype& x) const {
    return x;
  }
  __forceinline__
  __device__ Dtype Reduce(const Dtype& a, const Dtype& b) const {
    return a + b;
  }
  __forceinline__
  __device__ Dtype PostReduce(const Dtype& x, const Dtype& output) const {
    return x;
  }
};

template<typename Dtype>
struct TransReduceOp<MAX, Dtype> {
  __forceinline__
  __device__ Dtype InitValue() const {
    return sizeof(Dtype) == sizeof(float) ? -CUDART_INF_F : -CUDART_INF;
  }
  __forceinline__
  __device__ Dtype Transform(const Dtype& x) const {
    return x;
  }
  __forceinline__
  __device__ Dtype Reduce(const Dtype& a, const Dtype& b) const {
    return fmax(a, b);
  }
  __forceinline__
  __device__ Dtype PostReduce(const Dtype& x, const Dtype& output) const {
    return x;
  }
};

template<typename Dtype>
struct TransReduceOp<MIN, Dtype> {
  __forceinline__
  __device__ Dtype InitValue() const {
    return sizeof(Dtype) == sizeof(float) ? CUDART_INF_F : CUDART_INF;
  }
  __forceinline__
  __device__ Dtype Transform(const Dtype& x) const {
    return x;
  }
  __forceinline__
  __device__ Dtype Reduce(const Dtype& a, const Dtype& b) const {
    return min(a, b);
  }
  __forceinline__
  __device__ Dtype PostReduce(const Dtype& x, const Dtype& output) const {
    return x;
  }
};

template<EnumTransformReduce TransReduceType, typename Dtype>
__global__
static void _vec_transform_reduce(const int dim,const Dtype* src, Dtype* dst,
  const TransReduceOp<TransReduceType, Dtype> op) {
  
  __shared__ Dtype sdata[CU1DBLOCK];
  Dtype tdata = op.InitValue();

  const int tid = threadIdx.x;
  const int vec_len = dim;
  const int grid_stride = gridDim.x * blockDim.x;
  int i = (blockIdx.x * blockDim.x + tid);
  
  // Grid reduce. Loop over the whole vector v.
  for (; i < vec_len; i += grid_stride) {
    tdata = op.Reduce(tdata, op.Transform(src[i]));
  }
  
  sdata[tid] = tdata;
  __syncthreads();

  // Tree reduce
# pragma unroll
  for (int shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
    if (tid < shift) {
      sdata[tid] = op.Reduce(sdata[tid], sdata[tid + shift]);
    }
    __syncthreads();
  }

  // Reduce last warp.
  if (tid < warpSize) {
    for (int shift = warpSize; shift > 0; shift >>= 1) {
      sdata[tid] = op.Reduce(sdata[tid], sdata[tid + shift]);
#ifdef __HIP_PLATFORM_AMD__
      __threadfence_block();
#endif
    }
  }
  
  // Output to vector dst.
  if (tid == 0)
    dst[blockIdx.x] = op.PostReduce(sdata[0], dst[blockIdx.x]);
}

template<EnumTransformReduce TransReduceType, typename Dtype>
__global__
static void _vec_transform_reduce_inplace(const int dim,Dtype* data,
  const TransReduceOp<TransReduceType, Dtype> op) {
  
  __shared__ Dtype sdata[CU1DBLOCK];
  Dtype tdata = op.InitValue();

  const int tid = threadIdx.x;
  const int vec_len = dim;
  const int grid_stride = gridDim.x * blockDim.x;
  int i = (blockIdx.x * blockDim.x + tid);
  
  // Grid reduce. Loop over the whole vector v.
  for (; i < vec_len; i += grid_stride) {
    tdata = op.Reduce(tdata, op.Transform(data[i]));
    data[i]=0;
  }
  
  sdata[tid] = tdata;
  __syncthreads();

  // Tree reduce
# pragma unroll
  for (int shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
    if (tid < shift) {
      sdata[tid] = op.Reduce(sdata[tid], sdata[tid + shift]);
    }
    __syncthreads();
  }

  // Reduce last warp.
  if (tid < warpSize) {
    for (int shift = warpSize; shift > 0; shift >>= 1) {
      sdata[tid] = op.Reduce(sdata[tid], sdata[tid + shift]);
#ifdef __HIP_PLATFORM_AMD__
      __threadfence_block();
#endif
    }
  }
  
  // Output to vector dst.
  if (tid == 0){
    data[blockIdx.x] = op.PostReduce(sdata[0], data[blockIdx.x]);
  }
}

template<EnumTransformReduce TransReduceType, typename Dtype>
__global__ static void _single_reduce(const int dim, Dtype* dst,
  const TransReduceOp<TransReduceType, Dtype> op){
  for(int i = 1; i < dim; i++){
    dst[0] = op.Reduce(dst[0], dst[i]);
    dst[i] = 0;
  }
}

template<typename Dtype>
void cuda_set(const int N, Dtype* data, Dtype val){
  _set<<<NNDCT_GET_BLOCKS(N), NNDCT_CUDA_NUM_THREADS>>>(
    N, data, val);
}
template 
void cuda_set<float>(const int N, float* data, float val);
template 
void cuda_set<double>(const int N, double* data, double val);
  

template<typename Dtype>
void cuda_scale_inplace(const int N, Dtype* data, Dtype scale){
  _scale_inplace<<<NNDCT_GET_BLOCKS(N), NNDCT_CUDA_NUM_THREADS>>>(
    N, data, scale);
}
template
void cuda_scale_inplace<float>(const int N, float* data, float scale);
template
void cuda_scale_inplace<double>(const int N, double* data, double scale);


template<typename Dtype>
void cuda_scale(const int N, const Dtype* src, Dtype* dst, Dtype scale){
  _scale<<<NNDCT_GET_BLOCKS(N), NNDCT_CUDA_NUM_THREADS>>>(
    N, src, dst, scale);
}
template
void cuda_scale<float>(const int N, const float* src, float* dst, float scale);
template
void cuda_scale<double>(const int N, const double* src, double* dst, double scale);


template<typename Dtype>
void cuda_pow(const int N, Dtype* data, Dtype pow){
  _pow<<<NNDCT_GET_BLOCKS(N), NNDCT_CUDA_NUM_THREADS>>>(
    N, data, pow);
}
template
void cuda_pow<float>(const int N, float* data, float pow);
template
void cuda_pow<double>(const int N, double* data, double pow);


template<typename Dtype>
void cuda_max(const int N, const Dtype* src, Dtype* dst){
  int dimGrid=NNDCT_GET_BLOCKS1D(N);
  _vec_transform_reduce<<<dimGrid, CU1DBLOCK>>>(
    N, src, dst, TransReduceOp<MAX, Dtype>());

  _single_reduce<<<1, 1>>>(
    dimGrid, dst, TransReduceOp<MAX, Dtype>());
}
template
void cuda_max<float>(const int N, const float* src, float* dst);
template
void cuda_max<double>(const int N, const double* src, double* dst);


template<typename Dtype>
void cuda_min(const int N, const Dtype* src, Dtype* dst){
  int dimGrid=NNDCT_GET_BLOCKS1D(N);
  _vec_transform_reduce<<<dimGrid, CU1DBLOCK>>>(
    N, src, dst, TransReduceOp<MIN, Dtype>());

  _single_reduce<<<1, 1>>>(
    dimGrid, dst, TransReduceOp<MIN, Dtype>());
}
template
void cuda_min<float>(const int N, const float* src, float* dst);
template
void cuda_min<double>(const int N, const double* src, double* dst);


template<typename Dtype>
void cuda_sum(const int N, const Dtype* src, Dtype* dst){
  int dimGrid=NNDCT_GET_BLOCKS1D(N);
  _vec_transform_reduce<<<dimGrid,CU1DBLOCK>>>(
    N, src, dst, TransReduceOp<SUM, Dtype>());

  _single_reduce<<<1, 1>>>(
    dimGrid, dst, TransReduceOp<SUM, Dtype>());
}
template
void cuda_sum<float>(const int N, const float* src, float* dst);
template
void cuda_sum<double>(const int N, const double* src, double* dst);


template<typename Dtype>
void cuda_sum_inplace(const int N, Dtype* data){
  int dimGrid = NNDCT_GET_BLOCKS1D(N);
  _vec_transform_reduce_inplace<<<dimGrid, CU1DBLOCK>>>(
    N, data, TransReduceOp<SUM, Dtype>());

  _single_reduce<<<1, 1>>>(
    dimGrid, data, TransReduceOp<SUM, Dtype>());
}
template
void cuda_sum_inplace<float>(const int N, float* data);
template
void cuda_sum_inplace<double>(const int N, double* data);


template<typename Dtype>
void cuda_sub(const int N, const Dtype* src, Dtype* dst){
  _sub<<<NNDCT_GET_BLOCKS(N), NNDCT_CUDA_NUM_THREADS>>>(
      N, src, dst);
}
template
void cuda_sub<float>(const int N, const float* src, float* dst);
template
void cuda_sub<double>(const int N, const double* src, double* dst);

