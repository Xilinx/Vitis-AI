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

//#include <mutex>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/quantize.hpp"
#include <float.h>

namespace caffe {

template <typename Dtype>
__global__ void gpu_fix_kernel1(const int n, const Dtype *x, Dtype *y,
                               Dtype step, Dtype lb, Dtype ub) {
  CUDA_KERNEL_LOOP(i, n) { y[i] = min(max(round(x[i] / step) * step, lb), ub); }
}

// sigmoid kernel: y = sigmoid(x)
template <typename Dtype>
__global__ void gpu_sigmoid_kernel(const int n, const Dtype *x, Dtype *y) {
  CUDA_KERNEL_LOOP(i, n) { y[i] = 1. / (1. + exp(-x[i])); }
}

template <typename Dtype>
__global__ void gpu_fix_kernel2(const int n, const Dtype *x, Dtype *y,
                               Dtype step, Dtype lb, Dtype ub) {
  CUDA_KERNEL_LOOP(i, n) { 
    Dtype tmp = x[i] / step;
    // simulate DPU where to save hardware resource
    if ( tmp < 0 && ( tmp - floor( tmp ) ) == 0.5 )
      tmp = ceil( tmp );
    else
      tmp = round( tmp );
    y[i] = min(max(tmp * step, lb), ub); 
  }
}

template <typename Dtype>
void caffe_gpu_fix(const int n, const Dtype *x, Dtype *y, const int bit_width,
                   const int p) {
  Dtype step = std::pow(Dtype(2), -p);
  Dtype lower_bound = -std::pow(Dtype(2), bit_width - 1) * step;
  Dtype upper_bound = std::pow(Dtype(2), bit_width - 1) * step - step;
  gpu_fix_kernel1<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, x, y, step, lower_bound, upper_bound);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void caffe_gpu_top_fix(const int n, const Dtype *x, Dtype *y, const int bit_width,
                   const int p) {
  Dtype step = std::pow(Dtype(2), -p);
  Dtype lower_bound = -std::pow(Dtype(2), bit_width - 1) * step;
  Dtype upper_bound = std::pow(Dtype(2), bit_width - 1) * step - step;
  gpu_fix_kernel2<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, x, y, step, lower_bound, upper_bound);
  CUDA_POST_KERNEL_CHECK;
}

template void caffe_gpu_fix<float>(const int n, const float *x, float *y,
                                   const int bit_width, const int p);
template void caffe_gpu_fix<double>(const int n, const double *x, double *y,
                                    const int bit_width, const int p);
template void caffe_gpu_top_fix<float>(const int n, const float *x, float *y,
                                   const int bit_width, const int p);
template void caffe_gpu_top_fix<double>(const int n, const double *x, double *y,
                                    const int bit_width, const int p);

// Overflow: minimize fix pos in terms of all weights and data do not overflow
template <typename Dtype>
Dtype caffe_gpu_fix_pos_overflow(const int n, const Dtype *x,
                                 const int bit_width) {
  // Use half of step as a guard
  Dtype fix_lb = -std::pow(2, bit_width - 1) - 0.5;
  Dtype fix_ub = std::pow(2, bit_width - 1) - 0.5;

  // Dynamic range [min, max]
  // Find min and max value in GPU
  auto min_max = thrust::minmax_element(thrust::device, x, x + n);

  // Copy to Host
  Dtype x_min, x_max;
  cudaMemcpy(&x_min, min_max.first, sizeof(Dtype), cudaMemcpyDeviceToHost);
  cudaMemcpy(&x_max, min_max.second, sizeof(Dtype), cudaMemcpyDeviceToHost);

  Dtype step = std::max(x_min / fix_lb, x_max / fix_ub);
  if (step == 0) {
    return SHRT_MAX;
  } else if(isnan(step)) {
    return SHRT_MIN;
  }

  return std::log2(1 / step);
}

template float caffe_gpu_fix_pos_overflow<float>(const int n, const float *x,
                                                 const int bit_width);
template double caffe_gpu_fix_pos_overflow<double>(const int n, const double *x,
                                                   const int bit_width);

// Diff_S: minimize L2 norm of fixed weights/activation and float weights/activation
template <typename Dtype>
Dtype caffe_gpu_fix_pos_diffs(const int n, const Dtype *x, const int bit_width,
                          const int range) {
  // Calc search range for scale
  int max_scale;
  Dtype fix_lb = -std::pow(2, bit_width - 1) - 0.5;
  Dtype fix_ub = std::pow(2, bit_width - 1) - 0.5;
  auto min_max = thrust::minmax_element(thrust::device, x, x + n);
  // Copy to Host
  Dtype x_min, x_max;
  cudaMemcpy(&x_min, min_max.first, sizeof(Dtype), cudaMemcpyDeviceToHost);
  cudaMemcpy(&x_max, min_max.second, sizeof(Dtype), cudaMemcpyDeviceToHost);

  // Find max_scale
  Dtype step = std::max(x_min / fix_lb, x_max / fix_ub);
  if (step == 0) {
    return SHRT_MAX;
  } else if(isnan(step)) {
    return SHRT_MIN;
  } else {
    max_scale = std::floor(std::log2(1 / step));
  }

  // Find fix pos in range [max_scale + range , max_scale]
  Dtype final_scale;
  final_scale = max_scale;
  Dtype fixed_diff_min = FLT_MAX;
  Dtype *buffer;
  CUDA_CHECK(cudaMalloc((void **)&buffer, n * sizeof(Dtype)));
  /* CHECK_NOTNULL(buffer); */
  for (int scale = max_scale; scale < max_scale + range; scale++) {
    caffe_gpu_fix<Dtype>(n, x, buffer, bit_width, scale);
    caffe_gpu_sub<Dtype>(n, x, buffer, buffer);
    caffe_gpu_powx<Dtype>(n, buffer, 2, buffer);
    Dtype fixed_diff;
    caffe_gpu_asum(n, buffer, &fixed_diff);
    if (fixed_diff < fixed_diff_min) {
      final_scale = scale;
      fixed_diff_min = fixed_diff;
    }
  }
  CUDA_CHECK(cudaFree(buffer));
  return final_scale;
}

template float caffe_gpu_fix_pos_diffs<float>(const int n, const float *x,
                                          const int bit_width,
                                          const int range);
template double caffe_gpu_fix_pos_diffs<double>(const int n, const double *x,
                                            const int bit_width,
                                            const int range);

// Diff_S_Sigmoid: minimize L2 norm of sigmoid(weights/activation) between fixed and float
template <typename Dtype>
Dtype caffe_gpu_fix_pos_diffs_sigmoid(const int n, const Dtype *x, const int bit_width,
                          const int range) {
  // Calc search range for scale
  int max_scale;
  Dtype fix_lb = -std::pow(2, bit_width - 1) - 0.5;
  Dtype fix_ub = std::pow(2, bit_width - 1) - 0.5;
  auto min_max = thrust::minmax_element(thrust::device, x, x + n);
  // Copy to Host
  Dtype x_min, x_max;
  cudaMemcpy(&x_min, min_max.first, sizeof(Dtype), cudaMemcpyDeviceToHost);
  cudaMemcpy(&x_max, min_max.second, sizeof(Dtype), cudaMemcpyDeviceToHost);

  // Find max_scale
  Dtype step = std::max(x_min / fix_lb, x_max / fix_ub);
  if (step == 0)
    max_scale = 0;
  else
    max_scale = std::floor(std::log2(1 / step));

  // Find fix pos in range [max_scale + range , max_scale]
  Dtype final_scale;
  final_scale = max_scale;
  Dtype fixed_diff_min = FLT_MAX;
  Dtype *sigmoid_x, *buffer;
  CUDA_CHECK(cudaMalloc((void **)&sigmoid_x, n * sizeof(Dtype)));
  CUDA_CHECK(cudaMalloc((void **)&buffer, n * sizeof(Dtype)));
  gpu_sigmoid_kernel<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, x, sigmoid_x);
  CUDA_POST_KERNEL_CHECK;
  LOG(INFO) << "calib start";
  for (int scale = max_scale; scale < max_scale + range; scale++) {
    caffe_gpu_fix<Dtype>(n, x, buffer, bit_width, scale);
    gpu_sigmoid_kernel<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
        n, buffer, buffer);
    CUDA_POST_KERNEL_CHECK;
    caffe_gpu_sub<Dtype>(n, sigmoid_x, buffer, buffer);
    caffe_gpu_powx<Dtype>(n, buffer, 2, buffer);
    Dtype fixed_diff;
    caffe_gpu_asum(n, buffer, &fixed_diff);
    if (fixed_diff < fixed_diff_min) {
      final_scale = scale;
      fixed_diff_min = fixed_diff;
    }
  }
  CUDA_CHECK(cudaFree(sigmoid_x));
  CUDA_CHECK(cudaFree(buffer));
  return final_scale;
}

template float caffe_gpu_fix_pos_diffs_sigmoid<float>(const int n, const float *x,
                                          const int bit_width,
                                          const int range);
template double caffe_gpu_fix_pos_diffs_sigmoid<double>(const int n, const double *x,
                                            const int bit_width,
                                            const int range);

/*
template <typename Dtype>
static __global__ void overflow_kernel(const int n, Dtype upper_bound, Dtype
lower_bound, const Dtype* x, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index]=(x[index]<=upper_bound && x[index]>=lower_bound)?Dtype(0):Dtype(1);
  }
}

template <typename Dtype>
static bool test_overflow(const int n, Dtype upper_bound, Dtype lower_bound,
const Dtype* data, Dtype* buffer) {
  overflow_kernel<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n,
upper_bound, lower_bound, data, buffer);
  CUDA_POST_KERNEL_CHECK;
  Dtype asum;
  caffe_gpu_asum(n, buffer, &asum);
  return asum>Dtype(0.5);
}

template <typename Dtype>
void caffe_gpu_fix_overflow(const int n, const Dtype* x, Dtype* y, const int
bit_level, const int max_scale, const int min_scale, int& final_scale) {
        final_scale=std::max(std::min(final_scale, max_scale), min_scale);
        int search_length=max_scale-min_scale+1;
        if(search_length<2) {
                final_scale=min_scale;
        }
        else {
                Dtype* buffer=y;
                if(x==y) {
                        buffer=static_cast<Dtype*>(Caffe::GpuBuffer(n*sizeof(Dtype)));
                        CHECK_NOTNULL(buffer);
                }

                vector<Dtype> upper_bound(search_length);
                vector<Dtype> lower_bound(search_length);
                for(int i=0; i<search_length; i++) {
                        upper_bound[i]=std::pow(Dtype(2), i+min_scale);
                        lower_bound[i]=-upper_bound[i]-std::pow(Dtype(2),
i+min_scale-bit_level);
                }

                vector<bool> overflow(search_length);
                vector<bool> tested(search_length, false);

                bool found=false;
                overflow[final_scale-min_scale]=test_overflow(n,
upper_bound[final_scale-min_scale],
                                lower_bound[final_scale-min_scale], x, buffer);
                tested[final_scale-min_scale]=true;
                if(!overflow[final_scale-min_scale]) {
                        if(final_scale==min_scale) {
                                found=true;
                        }
                        else {
                                overflow[final_scale-min_scale-1]=test_overflow(n,
upper_bound[final_scale-min_scale-1],
                                                lower_bound[final_scale-min_scale-1],
x, buffer);
                                tested[final_scale-min_scale-1]=true;
                                if(overflow[final_scale-min_scale-1]) {
                                        found=true;
                                }
                        }
                }

                if(!found) {
                        overflow[0]=true;
                        tested[0]=true;
                        overflow[search_length-1]=false;
                        tested[search_length-1]=true;
                        int left=0;
                        int right=search_length-1;
                        for(;;) {
                                int middle=(left+right)/2;
                                if(!tested[middle]) {
                                        overflow[middle]=test_overflow(n,
upper_bound[middle], lower_bound[middle], x, buffer);
                                        tested[middle]=true;
                                }
                                if(!tested[middle+1]) {
                                        overflow[middle+1]=test_overflow(n,
upper_bound[middle+1], lower_bound[middle+1], x, buffer);
                                        tested[middle+1]=true;
                                }
                                if(overflow[middle] && !overflow[middle+1]) {
                                        final_scale=min_scale+middle+1;
                                        break;
                                }
                                else if(!overflow[middle]) {
                                        right=middle;
                                }
                                else {
                                        left=middle+1;
                                }
                        }
                }
        }
        caffe_gpu_fix(n, x, y, bit_level, final_scale);
}
template void caffe_gpu_fix_overflow<float>(const int n, const float* x, float*
y, const int bit_level, const int max_scale, const int min_scale, int&
final_scale);
template void caffe_gpu_fix_overflow<double>(const int n, const double* x,
double* y, const int bit_level, const int max_scale, const int min_scale, int&
final_scale);
*/

template <typename Dtype>
__global__ void gpu_scale_kernel(const int n, const Dtype *x, Dtype *y,
                               Dtype step ) {
  CUDA_KERNEL_LOOP(i, n) { y[i] = x[i] * step; }
}

template <typename Dtype>
void caffe_gpu_scale(const int n, const Dtype *x, Dtype *y, const int p) {
  Dtype step;
  if (p == SHRT_MAX) {
    step = 1;
  } else {
    step = std::pow(Dtype(2), p);
  }
  gpu_scale_kernel<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, x, y, step);

  CUDA_POST_KERNEL_CHECK;
}

template void caffe_gpu_scale<float>(const int n, const float *x, float *y, const int p);
template void caffe_gpu_scale<double>(const int n, const double *x, double *y, const int p);

template <typename Dtype>
__global__ void gpu_trunc_kernel(const int n, const Dtype *x, Dtype *y,
                               Dtype scale) {
  CUDA_KERNEL_LOOP(i, n) { y[i] = ( (int)(x[i] / scale) ) * scale; }
}

template <typename Dtype>
void caffe_gpu_trunc(const int n, const Dtype *x, Dtype *y, const int p) {
  Dtype scale = std::pow(Dtype(2), -p);
  gpu_trunc_kernel<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, x, y, scale);

  CUDA_POST_KERNEL_CHECK;
}

template void caffe_gpu_trunc<float>(const int n, const float *x, float *y, const int p);
template void caffe_gpu_trunc<double>(const int n, const double *x, double *y, const int p);

template <typename Dtype>
void caffe_pooling_scale(const int n, const Dtype *x, Dtype *y,  float scale) {
  gpu_scale_kernel<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, x, y, scale);

  CUDA_POST_KERNEL_CHECK;
}

template void caffe_pooling_scale<float>(const int n, const float *x, float *y, float scale);
template void caffe_pooling_scale<double>(const int n, const double *x, double *y, float scale);


} // namespace caffe
