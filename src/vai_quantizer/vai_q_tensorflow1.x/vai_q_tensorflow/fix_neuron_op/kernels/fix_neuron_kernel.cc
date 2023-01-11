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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>

#include "fix_neuron_ops.h"

namespace tensorflow {
namespace functor {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <typename T>
T quantize_kernel_cpu(const T x, const T step, const T lower_bound,
                      const T upper_bound) {
  return std::fmin(std::fmax(std::round(x / step) * step, lower_bound),
                   upper_bound);
}

template <typename T>
T quantize_kernel_cpu_dpu(const T x, const T step, const T lower_bound,
                          const T upper_bound) {
  T tmp = x / step;
  // Simulate DPU where to save hardware resource
  if (tmp < 0 && (tmp - std::floor(tmp)) == 0.5) {
    tmp = std::ceil(tmp);
  } else {
    tmp = std::round(tmp);
  }
  return std::fmin(std::fmax(tmp * step, lower_bound), upper_bound);
}

template <typename T>
void dp_quantize_cpu(const int n, const T *x, T *y, const int bit_width,
                     const int pos, const int mode) {
  T step = std::pow(T(2), -pos);
  T lower_bound = -std::pow(T(2), bit_width - 1) * step;
  T upper_bound = std::pow(T(2), bit_width - 1) * step - step;
  // default arch_type is [-128, 127]
  string arch_type = getenv("ARCH_TYPE") ? getenv("ARCH_TYPE") : "DEFAULT";
  if (arch_type == "DPUCADF8H" && mode == 1) {
    lower_bound += step;
  }
  // default dpu_accuracy is true
  bool dpu_accuracy =
      (!getenv("DPU_ACCURACY")) ||
      (getenv("DPU_ACCURACY") && std::atoi(getenv("DPU_ACCURACY")) == 1);
  for (auto i = 0; i < n; ++i) {
    if (mode != 1 || (!dpu_accuracy)) {
      // Do normal round for weights/biases and if DPU_ACCURACY is false
      y[i] = quantize_kernel_cpu(x[i], step, lower_bound, upper_bound);
    } else {
      // Do DPU round for activation if DPU_ACCURACY is true
      y[i] = quantize_kernel_cpu_dpu(x[i], step, lower_bound, upper_bound);
    }
  }
}

template <typename T>
T l2_diff_cpu(const int n, const T *x, const T *y, const int method,
              const int mode) {
  T diff = 0;
  if (method == 2 && mode == 2) {
    // diff += std::pow(abs((y[i] - x[i]) / (abs(x[i]) + 1e-9)), 2);
    // diff += std::pow(abs((y[i] - x[i]) / (abs(x[i]) + 1e-9)), 1);
    for (auto i = 0; i < n; ++i) {
      diff += std::pow(abs((y[i] - x[i]) / (abs(x[i]) + 1e-9)), 0.5);
    }
  } else {
    for (auto i = 0; i < n; ++i) {
      diff += std::pow(y[i] - x[i], 2);
    }
  }
  return diff;
}

// Quantize Pos(Overflow): maximize quantize pos in terms of that all input data
// does not overflow
template <typename T>
int dp_get_quantize_pos_overflow_cpu(const int n, const T *x,
                                     const int bit_width) {
  T x_min = std::numeric_limits<float>::max();
  T x_max = -std::numeric_limits<float>::max();
  for (auto i = 0; i < n; ++i) {
    x_min = std::min(x_min, x[i]);
    x_max = std::max(x_max, x[i]);
  }
  // Use 0.5 as a guard
  T lower_bound = -std::pow(T(2), bit_width - 1) - 0.5;
  T upper_bound = std::pow(T(2), bit_width - 1) - 0.5;
  T step = std::max(x_min / lower_bound, x_max / upper_bound);
  if (step == 0) {
    // Set pos to 127(max of uint8) for all zero
    return 127;
  } else {
    VLOG(1) << "CPU OVERFLOW: min: " << x_min << " max: " << x_max
            << " pos: " << std::floor(std::log2(1 / step));
    return std::floor(std::log2(1 / step));
  }
}

// Quantize Pos(Diff_S): find position that maximizes L2 Difference between
// float and quantized data after quantization.
template <typename T>
int dp_get_quantize_pos_diffs_cpu(const int n, const T *x, const int bit_width,
                                  const int method, const int mode,
                                  const int range = 5) {
  int pos_overflow = dp_get_quantize_pos_overflow_cpu(n, x, bit_width);
  if (pos_overflow == 127) {
    // Set pos to 127(max of uint8) for all zero
    return 127;
  }
  int pos_diffs = pos_overflow;
  T diff_min = std::numeric_limits<float>::max();

  T *tmp_y = new T[sizeof(T) * n];
  for (auto i = 0; i < range; i++) {
    dp_quantize_cpu(n, x, tmp_y, bit_width, pos_overflow + i, mode);
    T diff = l2_diff_cpu(n, x, tmp_y, method, mode);
    if (diff < diff_min) {
      diff_min = diff;
      pos_diffs = pos_overflow + i;
    }
  }

  VLOG(1) << "CPU DIFFS: " << pos_diffs << " OVERFLOW: " << pos_overflow;
  delete[] tmp_y;
  return pos_diffs;
}

// This is the CPU kernel.
template <typename T> struct FixNeuronFunctor<CPUDevice, T> {
  void operator()(const CPUDevice &d, const T *input_tensor,
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
            dp_get_quantize_pos_overflow_cpu(size, input_tensor, bit_width);
      } else if (method == 1 || method == 2) {
        current_pos = dp_get_quantize_pos_diffs_cpu(size, input_tensor,
                                                    bit_width, method, mode);
      } else {
        LOG(FATAL) << "Invalid quantize method: " << method;
      }
      dp_quantize_cpu(size, input_tensor, output_tensor, bit_width, current_pos,
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
      dp_quantize_cpu(size, input_tensor, output_tensor, bit_width, saved_pos,
                      mode);

    } else {
      LOG(FATAL) << "Invalid phase for FixNeuron Op: " << phase;
    }
  }
};

template struct FixNeuronFunctor<CPUDevice, float>;

} // namespace functor
} // namespace tensorflow
