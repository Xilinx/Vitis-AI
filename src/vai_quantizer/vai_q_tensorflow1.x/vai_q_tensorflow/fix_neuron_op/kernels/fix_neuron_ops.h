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

#ifndef FIX_NEURON_OPS_H_
#define FIX_NEURON_OPS_H_
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace functor {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// kernel
template <typename Device, typename T>
struct FixNeuronFunctor {
  void operator()(const Device& d, const T* input_tensor, const int& bit_width,
                  const int& method, const int& mode, const int& phase,
                  int& quantize_pos, int& iter, std::map<int, int>& pos_hist,
                  T* output_tensor, const int& size);
};

}  // namespace functor
}  // namespace tensorflow
#endif  // FIX_NEURON_OPS_H_
