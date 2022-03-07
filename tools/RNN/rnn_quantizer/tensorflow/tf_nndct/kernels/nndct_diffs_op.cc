

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

#define EIGEN_USE_GPU

using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
using tensorflow::shape_inference::InferenceContext;


namespace functor {
// Functor used by FixNeuronV2Op to do the computations.
template <typename Device, typename T>
struct DiffSFunctor;

template <typename T>
struct DiffSFunctor<CPUDevice,T> {
  void operator()(int N,
                  const T* input,
                  T* buffer,
                  T* output,
                  int bitwidth,
                  int range,
                  int method){
    cpu_diff_S(N,
                input,
                buffer,
                output,
                bitwidth,
                range,
                method);
    // printf("NNDCT-warning: Test TF NNDCT support CPU flow!!! From nndct diffs op!\n");
    // fflush(stdout);
  }
};

template <typename T>
struct DiffSFunctor<GPUDevice,T> {
  void operator()(int N, const T* input, T* buffer, T* output, int bitwidth,
      int range, int method) {
    cuda_diff_S(N,
                input,
                buffer,
                output,
                bitwidth,
                range,
                method);
  }
};


}//namespace functor

REGISTER_OP("NndctDiffS")
    .Attr("T:{float,double}=DT_FLOAT")
    .Attr("bitwidth: int=8")
    .Attr("range: int=5")
    .Attr("method: int=4")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->MakeShape({}));
      return Status::OK();
    });

template <typename Device, typename T>
class NndctDiffSOp : public OpKernel {
 public:
  virtual ~NndctDiffSOp(){}

  explicit NndctDiffSOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("bitwidth", &bitwidth_));
    OP_REQUIRES_OK(context, context->GetAttr("range", &range_));
    OP_REQUIRES_OK(context, context->GetAttr("method", &method_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    Tensor* output = nullptr;
    Tensor buffer;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &output));
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
        input.shape(), &buffer));
    const T* input_data = input.flat<T>().data();
    T* buffer_data = buffer.flat<T>().data();
    T* output_data = output->flat<T>().data();

#ifdef QUANT_DEBUG
    printf("\n......Diffs OP conext i/o data: %p %p count: %ld %ld, dims: %d dim --",
            input_data,
            output_data,
            (long int)(input.NumElements()),
            (long int)(output->NumElements()),
            input.dims());
    fflush(stdout);
    if ( input.dims() > 0 ) {
      for ( int i = 0; i < input.dims(); ++i )
        printf( " %d", (int)(input.dim_size(i)) ); fflush(stdout);
    }
    printf( " --\n" );fflush(stdout);
#endif // QUANT_DEBUG

    functor::DiffSFunctor<Device, T>()(input.NumElements(),
                                       input_data,
                                       buffer_data,
                                       output_data,
                                       bitwidth_,
                                       range_,
                                       method_);
  }

  private:
    int bitwidth_;
    int range_;
    int method_;
};

#define REGISTER_CPU(T)                 \
  REGISTER_KERNEL_BUILDER(              \
      Name("NndctDiffS").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      NndctDiffSOp<CPUDevice,T>);

REGISTER_CPU(float);
REGISTER_CPU(double);

#define REGISTER_GPU(T)                 \
  REGISTER_KERNEL_BUILDER(              \
      Name("NndctDiffS").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      NndctDiffSOp<GPUDevice,T>);

REGISTER_GPU(float);
REGISTER_GPU(double);

}  // namespace nndct
