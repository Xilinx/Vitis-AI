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

#include <cmath>
#include <cstdlib>
#include <fstream>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/strings/str_util.h"

#include "fix_neuron_ops.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
using functor::FixNeuronFunctor;

void save_or_update_quantize_info(const string &op_name, const int &bit_width,
                                  const int &pos, const string &output_dir) {
  string filename;
  filename =
      output_dir + "/temp/" + str_util::StringReplace(op_name, "/", "_", true);
  std::ofstream ofile(filename);
  if (!ofile.is_open()) {
    LOG(FATAL) << "Cannot open file: " << filename << " for op: " << op_name;
  }
  ofile << op_name << " " << bit_width << " " << pos << std::endl;
  // LOG(INFO) << "Saving file:" << filename << " pos: " << pos;
  ofile.close();
}

template <typename Device, typename T> class FixNeuronOp : public OpKernel {
public:
  explicit FixNeuronOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("bit_width", &bit_width_));
    OP_REQUIRES_OK(context, context->GetAttr("method", &method_));
    OP_REQUIRES_OK(context, context->GetAttr("mode", &mode_));
    OP_REQUIRES_OK(context, context->GetAttr("phase", &phase_));
    OP_REQUIRES_OK(context, context->GetAttr("output_dir", &output_dir_));
    if (phase_ == 1) {
      // load pos from file if phase is Evaluation
      OP_REQUIRES_OK(context, context->GetAttr("quantize_pos", &quantize_pos_));
    }
    iter_ = 0;
  }

  ~FixNeuronOp() {}

  void Compute(OpKernelContext *context) override {
    const Tensor &input_tensor = context->input(0);
    Tensor *output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    FixNeuronFunctor<Device, T>()(
        context->eigen_device<Device>(), input_tensor.flat<T>().data(),
        bit_width_, method_, mode_, phase_, quantize_pos_, iter_, pos_hist_,
        output_tensor->flat<T>().data(), input_tensor.flat<T>().size());
    // save pos to file if phase is Calibration or Training
    if (phase_ == 0 || phase_ == 2) {
      save_or_update_quantize_info(this->name(), bit_width_, quantize_pos_,
                                   output_dir_);
    }
  }

private:
  int bit_width_;
  // method
  // 0: overflow
  // 1: diffs
  // 2: diffs with depthwise strategy
  int method_;
  int quantize_pos_;
  int iter_;
  std::map<int, int> pos_hist_;
  // Mode
  // 0: For normal weights/biases
  // 1: For activation
  // 2: For depthwise weights
  int mode_;
  // Phase
  // 0: Calibration
  // 1: Evaluation
  // 2: Training
  int phase_;
  string output_dir_;
};

// Register the CPU kernels
#define REGISTER_CPU(T)                                                        \
  REGISTER_KERNEL_BUILDER(Name("FixNeuron").Device(DEVICE_CPU),                \
                          FixNeuronOp<CPUDevice, T>);
REGISTER_CPU(float);
#undef REGISTER_CPU

#if GOOGLE_CUDA
// register GPU kernel.
#define REGISTER_GPU(T)                                                        \
  REGISTER_KERNEL_BUILDER(Name("FixNeuron").Device(DEVICE_GPU),                \
                          FixNeuronOp<GPUDevice, T>);
REGISTER_GPU(float);
#undef REGISTER_GPU
#endif // GOOGLE_CUDA

} // namespace tensorflow
