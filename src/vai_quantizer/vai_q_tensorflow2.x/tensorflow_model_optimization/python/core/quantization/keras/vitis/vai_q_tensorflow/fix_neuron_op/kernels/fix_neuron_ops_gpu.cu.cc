/*Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.*/

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <cmath>
#include <iostream>
#include <map>

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "fix_neuron_ops.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace functor {

using GPUDevice = Eigen::GpuDevice;

#define CHECK_CUBLAS_ERROR(state)                                              \
  if (CUBLAS_STATUS_SUCCESS != state)                                          \
    printf("CUBLAS ERROR state %d in file %s at line %d.\n", state, __FILE__,  \
           __LINE__);
#define CHECK_CUDA_MALLOC(state)                                               \
  if (cudaSuccess != state)                                                    \
    printf(                                                                    \
        "Fail to malloc gpu memory, please check your GPU. Note: batch_size "  \
        "will affect the GPU memory occupation, maybe you can decrease "       \
        "batch_size and try again.");

template <typename T>
__global__ void quantize_kernel_gpu(const int n, const T *x, T *y, const T step,
                                    const T lower_bound, const T upper_bound) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    y[i] = min(max(round(x[i] / step) * step, lower_bound), upper_bound);
  }
}

template <typename T>
__global__ void quantize_kernel_gpu_dpu(const int n, const T *x, T *y,
                                        const T step, const T lower_bound,
                                        const T upper_bound) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    T tmp = x[i] / step;
    // Simulate DPU where to save hardware resource
    if (tmp < 0 && (tmp - floor(tmp)) == 0.5) {
      tmp = ceil(tmp);
    } else {
      tmp = round(tmp);
    }
    y[i] = min(max(tmp * step, lower_bound), upper_bound);
  }
}

template <typename T>
__global__ void l2_diff_kernel_gpu(const int n, const T *x, const T *y, T *res,
                                   const int method, const int mode) {
  if (method == 2 && mode == 2) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
      // res[i] = pow((y[i] - x[i]) / (abs(x[i]) + 1e-9), 2);
      // res[i] = abs((y[i] - x[i]) / (abs(x[i]) + 1e-9));
      res[i] = pow(abs((y[i] - x[i]) / (abs(x[i]) + 1e-9)), 0.5);
    }
  } else {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
      res[i] = pow(y[i] - x[i], 2);
    }
  }
}

template <typename T>
void dp_quantize_gpu(const int n, const T *x, T *y, const int bit_width,
                     const int pos, const int mode) {
  T step = std::pow(T(2), -pos);
  T lower_bound = -std::pow(T(2), bit_width - 1) * step;
  T upper_bound = std::pow(T(2), bit_width - 1) * step - step;
  // default arch_type is [-128, 127]
  string arch_type = getenv("ARCH_TYPE") ? getenv("ARCH_TYPE") : "DEFAULT";
  if (arch_type == "DPUCADF8H" && mode == 1) {
    lower_bound += step;
  }
  int block_count = 1024;
  int thread_per_block = 20;
  // default dpu_accuracy is true
  bool dpu_accuracy =
      (!getenv("DPU_ACCURACY")) ||
      (getenv("DPU_ACCURACY") && std::atoi(getenv("DPU_ACCURACY")) == 1);
  if (mode != 1 || (!dpu_accuracy)) {
    // Do normal round for weights/biases and if DPU_ACCURACY is false
    quantize_kernel_gpu<T><<<block_count, thread_per_block>>>(
        n, x, y, step, lower_bound, upper_bound);
  } else {
    // Do DPU round for activation if DPU_ACCURACY is true
    quantize_kernel_gpu_dpu<T><<<block_count, thread_per_block>>>(
        n, x, y, step, lower_bound, upper_bound);
  }
}

template <typename T>
T l2_diff_gpu(const int n, const T *x, const T *y, const int method,
              const int mode) {
  cublasHandle_t handle;
  CHECK_CUBLAS_ERROR(cublasCreate(&handle));
  int block_count = 1024;
  int thread_per_block = 20;
  T *res_gpu;
  CHECK_CUDA_MALLOC(cudaMalloc((void **)&res_gpu, n * sizeof(T)));
  l2_diff_kernel_gpu<T>
      <<<block_count, thread_per_block>>>(n, x, y, res_gpu, method, mode);
  float sum;
  CHECK_CUBLAS_ERROR(cublasSasum(handle, n, res_gpu, 1, &sum));
  cublasDestroy(handle);
  cudaFree(res_gpu);
  return sum;
}

// Quantize Pos(Overflow): maximize quantize pos in terms of that all input data
// does not overflow
template <typename T>
int dp_get_quantize_pos_overflow_gpu(const int n, const T *x,
                                     const int bit_width) {
  T x_min, x_max;
  auto min_max = thrust::minmax_element(thrust::device, x, x + n);
  cudaMemcpy(&x_min, min_max.first, sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(&x_max, min_max.second, sizeof(T), cudaMemcpyDeviceToHost);
  // Use 0.5 as a guard
  T lower_bound = -std::pow(T(2), bit_width - 1) - 0.5;
  T upper_bound = std::pow(T(2), bit_width - 1) - 0.5;
  T step = std::max(x_min / lower_bound, x_max / upper_bound);
  if (step == 0) {
    // Set pos to 127(max of uint8) for all zero
    return 127;
  } else {
    VLOG(1) << "GPU OVERFLOW: min: " << x_min << " max: " << x_max
            << " pos: " << std::floor(std::log2(1 / step));
    return std::floor(std::log2(1 / step));
  }
}

// Quantize Pos(Diff_S): find position that maximizes L2 Difference between
// float and quantized data after quantization.
template <typename T>
int dp_get_quantize_pos_diffs_gpu(const int n, const T *x, const int bit_width,
                                  const int method, const int mode,
                                  const int range = 5) {
  int pos_overflow = dp_get_quantize_pos_overflow_gpu(n, x, bit_width);
  if (pos_overflow == 127) {
    // Set pos to 127(max of uint8) for all zero
    return 127;
  }
  int pos_diffs = pos_overflow;
  T diff_min = std::numeric_limits<float>::max();

  T *tmp_y;
  cudaMalloc((void **)&tmp_y, n * sizeof(T));
  for (auto i = 0; i < range; i++) {
    dp_quantize_gpu(n, x, tmp_y, bit_width, pos_overflow + i, mode);
    T diff = l2_diff_gpu(n, x, tmp_y, method, mode);
    if (diff < diff_min) {
      diff_min = diff;
      pos_diffs = pos_overflow + i;
    }
  }
  cudaFree(tmp_y);

  VLOG(1) << "GPU DIFFS: " << pos_diffs << " OVERFLOW: " << pos_overflow;
  return pos_diffs;
}

// This is the GPU kernel
template <typename T> struct FixNeuronFunctor<GPUDevice, T> {
  void operator()(const GPUDevice &d, const T *input_tensor,
                  const int &bit_width, const int &method, const int &mode,
                  const int &phase, int &saved_pos, int &iter,
                  std::map<int, int> &pos_hist, T *output_tensor,
                  const int &size) {
    // Training & Calibration Phase
    // Fisrt get the current_pos, then quantize activation/weights using
    // current_pos
    // Update saved pos and pos_hist for Calibration Phase
    if (phase == 2 || phase == 0) {
      int current_pos;
      if (method == 0) {
        current_pos =
            dp_get_quantize_pos_overflow_gpu(size, input_tensor, bit_width);
      } else if (method == 1 || method == 2) {
        current_pos = dp_get_quantize_pos_diffs_gpu(size, input_tensor,
                                                    bit_width, method, mode);
      } else {
        LOG(FATAL) << "Invalid quantize method: " << method;
      }
      dp_quantize_gpu(size, input_tensor, output_tensor, bit_width, current_pos,
                      mode);
      iter++;
      pos_hist[current_pos]++;
      auto max_ele = std::max_element(
          pos_hist.begin(), pos_hist.end(),
          [](const std::pair<int, int> &p1, const std::pair<int, int> &p2) {
            return p1.second < p2.second;
          });
      saved_pos = max_ele->first;

      // Evaluation Phase
      // quantize activation/weights using saved_pos
    } else if (phase == 1) {
      dp_quantize_gpu(size, input_tensor, output_tensor, bit_width, saved_pos,
                      mode);

    } else {
      LOG(FATAL) << "Invalid phase for FixNeuron Op: " << phase;
    }
  }
};

// // An instantiate.
template struct FixNeuronFunctor<GPUDevice, float>;

} // namespace functor
} // namespace tensorflow
#endif // GOOGLE_CUDA
