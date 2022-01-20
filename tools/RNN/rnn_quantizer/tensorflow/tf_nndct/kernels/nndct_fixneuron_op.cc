

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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"

#include "nndct_fix_kernels.h"
#include "nndct_fix_kernels_cpu.h"
#include "nndct_cuda_math.h"
#include "nndct_cpu_math.h"

namespace nndct {

using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace functor {
// Functor used by FixNeuronV2Op to do the computations.
template <typename Device, typename T>
struct FixNeuronV2;

template <typename T>
struct FixNeuronV2<CPUDevice,T> {
  void operator()(OpKernelContext* ctx,
                  const Tensor* Tinput,
                  Tensor* Toutput,
                  int valmax,
                  T valamp,
                  int method){
    const T* input = Tinput->flat<T>().data();
    T* output = Toutput->flat<T>().data();
    cpu_fix_neuron_v2(Tinput->NumElements(),
                       input,
                       output,
                       valmax,
                       valamp,
                       1, // keep_scale
                       method);
    // printf("NNDCT-warning: Test TF NNDCT support CPU flow!!! From nndct fixneuron op!\n");
    // fflush(stdout);
  }
};

template <typename T>
struct FixNeuronV2<GPUDevice,T> {
  void operator()(OpKernelContext* ctx,
                  const Tensor* Tinput,
                  Tensor* Toutput,
                  int valmax,
                  T valamp,
                  int method){
    const T* input = Tinput->flat<T>().data();
    T* output = Toutput->flat<T>().data();
    cuda_fix_neuron_v2(Tinput->NumElements(),
                       input,
                       output,
                       valmax,
                       valamp,
                       1, // keep_scale
                       method);
  }
};

}//namespace functor

REGISTER_OP("NndctFixNeuronV2")
    .Attr("T:{float,double}=DT_FLOAT")
    .Attr("valmax: int=128")
    .Attr("valamp: float=1.0")
    .Attr("method: int=4")
    .Input("input: T")
    .Output("result: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

template <typename Device, typename T>
class FixNeuronV2Op : public OpKernel {
 public:
  virtual ~FixNeuronV2Op(){}

  explicit FixNeuronV2Op(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("valmax", &valmax_));
    OP_REQUIRES_OK(context, context->GetAttr("valamp", &valamp_));
    OP_REQUIRES_OK(context, context->GetAttr("method", &method_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,context->allocate_output(0, input.shape(), &output));

    functor::FixNeuronV2<Device, T>()(context,
                                      &input,
                                      output,
                                      valmax_,
                                      valamp_,
                                      method_);
  }

  private:
    int method_;
    int valmax_;
    float valamp_;
};

#define REGISTER_CPU(T)                 \
  REGISTER_KERNEL_BUILDER(              \
      Name("NndctFixNeuronV2").Device(DEVICE_CPU).TypeConstraint<T>("T"),\
      FixNeuronV2Op<CPUDevice,T>);

REGISTER_CPU(float);
REGISTER_CPU(double);

#define REGISTER_GPU(T)                 \
  REGISTER_KERNEL_BUILDER(              \
      Name("NndctFixNeuronV2").Device(DEVICE_GPU).TypeConstraint<T>("T"),\
      FixNeuronV2Op<GPUDevice,T>);

REGISTER_GPU(float);
REGISTER_GPU(double);

}  // namespace nndct
