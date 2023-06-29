

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

#include "nndct_cuda_math.h"
#include "nndct_cpu_math.h"

namespace nndct {

using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
using tensorflow::shape_inference::InferenceContext;

namespace functor {
// Functor used by ScaleOp to do the computations.
template <typename Device, typename T>
struct ScaleFunctor;

template <typename T>
struct ScaleFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* ctx,
                  const Tensor* input,
                  Tensor* output,
                  float scale){
    const T* input_buffer = input->flat<T>().data();
    T* output_buffer = output->flat<T>().data();
#ifdef QUANT_DEBUG
    printf("\n......Scale OP conext i/o data: %p %p %p \
count: %ld %ld, dims: %d dim --",
            input_buffer,
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

    cpu_scale(input->NumElements(),
               input_buffer,
               output_buffer,
               (T)scale);
    // printf("NNDCT-warning: TF NNDCT does not support CPU flow yet!!! From scale op!\n");
    // fflush(stdout);
  }
};

template <typename T>
struct ScaleFunctor<GPUDevice,T> {
  void operator()(OpKernelContext* ctx,
                  const Tensor* input,
                  Tensor* output,
                  float scale) {
    const T* input_buffer = input->flat<T>().data();
    T* output_buffer = output->flat<T>().data();

#ifdef QUANT_DEBUG
    printf("\n......Scale OP conext i/o data: %p %p %p \
count: %ld %ld, dims: %d dim --",
            input_buffer,
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

    cuda_scale(input->NumElements(),
               input_buffer,
               output_buffer,
               (T)scale);
  }
};

}//namespace functor

REGISTER_OP("NndctScaleOp")
    .Attr("T:{float,double}=DT_FLOAT")
    .Attr("scale:float=1.0")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

template <typename Device, typename T>
class ScaleOp : public OpKernel {
 public:
  virtual ~ScaleOp(){}

  explicit ScaleOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("", &scale_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,context->allocate_output(0, input.shape(), &output));

    functor::ScaleFunctor<Device, T>()(
        context,
        &input,
        output,
        scale_);
  }

  private:
    float scale_;
};

#define REGISTER_CPU(T)                 \
  REGISTER_KERNEL_BUILDER(              \
      Name("NndctScaleOp").Device(DEVICE_CPU).TypeConstraint<T>("T"),\
      ScaleOp<CPUDevice,T>);

REGISTER_CPU(float);
REGISTER_CPU(double);

#define REGISTER_GPU(T)                 \
  REGISTER_KERNEL_BUILDER(              \
      Name("NndctScaleOp").Device(DEVICE_GPU).TypeConstraint<T>("T"),\
      ScaleOp<GPUDevice,T>);

REGISTER_GPU(float);
REGISTER_GPU(double);

}  // namespace nndct
