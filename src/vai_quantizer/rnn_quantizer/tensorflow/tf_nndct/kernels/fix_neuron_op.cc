

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

namespace nndct {

using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
using tensorflow::shape_inference::InferenceContext;

namespace functor {
// Functor used by FixNeuronOp to do the computations.
template <typename Device, typename T>
struct FixNeuronFunctor;

template <typename T>
struct FixNeuronFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* ctx,
                  const Tensor* input,
                  Tensor* output,
                  const Tensor* fp,
                  int bit_width,
                  int method){

    const T* input_buffer = input->flat<T>().data();
    const T* fp_buffer = fp->flat<T>().data();
    T* output_buffer = output->flat<T>().data();

#ifdef QUANT_DEBUG
    printf("\n......FixNeuron OP conext i/fp/o data: %p %p %p \
count: %ld %ld, dims: %d dim --",
            input_buffer,
            fp_buffer,
            output_buffer,
            (long int)(input->NumElements()),
            (long int)(output->NumElements()),
            input->dims());
    fflush(stdout);
    if ( input->dims() > 0 ) {
      for ( int i = 0; i < input->dims(); ++i )
        printf( " %d", (int)(input->dim_size(i)) ); fflush(stdout);
    }
    printf( " --\n" );fflush(stdout);
#endif // QUANT_DEBUG

    cpu_fix_neuron_v1(input->NumElements(),
                       input_buffer,
                       fp_buffer,
                       output_buffer,
                       1<<(bit_width-1),
                       1, // keep_scale
                       method);
    // printf("NNDCT-warning: Test TF NNDCT support CPU flow!!! From fix neuron op!\n");
    // fflush(stdout);
  }
};

template <typename T>
struct FixNeuronFunctor<GPUDevice,T> {
  void operator()(OpKernelContext* ctx,
                  const Tensor* input,
                  Tensor* output,
                  const Tensor* fp,
                  int bit_width,
                  int method) {
    const T* input_buffer = input->flat<T>().data();
    const T* fp_buffer = fp->flat<T>().data();
    T* output_buffer = output->flat<T>().data();

#ifdef QUANT_DEBUG
    printf("\n......FixNeuron OP conext i/fp/o data: %p %p %p \
count: %ld %ld, dims: %d dim --",
            input_buffer,
            fp_buffer,
            output_buffer,
            (long int)(input->NumElements()),
            (long int)(output->NumElements()),
            input->dims());
    fflush(stdout);
    if ( input->dims() > 0 ) {
      for ( int i = 0; i < input->dims(); ++i )
        printf( " %d", (int)(input->dim_size(i)) ); fflush(stdout);
    }
    printf( " --\n" );fflush(stdout);
#endif // QUANT_DEBUG

    cuda_fix_neuron_v1(input->NumElements(),
                       input_buffer,
                       fp_buffer,
                       output_buffer,
                       1<<(bit_width-1),
                       1, // keep_scale
                       method);
  }
};

}//namespace functor

REGISTER_OP("NndctFixNeuron")
    .Attr("T:{float,double}=DT_FLOAT")
    .Attr("bit_width: int=128")
    .Attr("method: int=4")
    .Input("input: T")
    .Input("fp_tensor: T")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

template <typename Device, typename T>
class FixNeuronOp : public OpKernel {
 public:
  virtual ~FixNeuronOp(){}

  explicit FixNeuronOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("bit_width", &bit_width_));
    OP_REQUIRES_OK(context, context->GetAttr("method", &method_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& fp_tensor = context->input(1);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,context->allocate_output(0, input.shape(), &output));

    functor::FixNeuronFunctor<Device, T>()(
        context,
        &input,
        output,
        &fp_tensor,
        bit_width_,
        method_);
  }

  private:
    int method_;
    int bit_width_;
};

#define REGISTER_CPU(T)                 \
  REGISTER_KERNEL_BUILDER(              \
      Name("NndctFixNeuron").Device(DEVICE_CPU).TypeConstraint<T>("T"),\
      FixNeuronOp<CPUDevice,T>);

REGISTER_CPU(float);
REGISTER_CPU(double);

#define REGISTER_GPU(T)                 \
  REGISTER_KERNEL_BUILDER(              \
      Name("NndctFixNeuron").Device(DEVICE_GPU).TypeConstraint<T>("T"),\
      FixNeuronOp<GPUDevice,T>);

REGISTER_GPU(float);
REGISTER_GPU(double);

}  // namespace nndct
