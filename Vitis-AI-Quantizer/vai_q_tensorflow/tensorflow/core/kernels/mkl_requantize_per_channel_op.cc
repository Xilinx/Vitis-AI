/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/array_ops.cc.

#ifdef INTEL_MKL
#define EIGEN_USE_THREADS
#include <math.h>

#include "mkldnn.hpp"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/meta_support.h"
#include "tensorflow/core/kernels/no_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename Toutput>
class MklRequantizePerChannelOp : public OpKernel {
 public:
  explicit MklRequantizePerChannelOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("out_type", &out_type_));
    OP_REQUIRES(ctx, out_type_ == DT_QINT8 || out_type_ == DT_QUINT8,
                errors::InvalidArgument(
                    "out_type must be qint8 or quint8, but got: ", out_type_));
  }
  virtual ~MklRequantizePerChannelOp() {}
  void Compute(OpKernelContext* ctx) override {
    try {
      const Tensor& input = ctx->input(kInputTensorIndex);
      const Tensor& input_min_vec = ctx->input(kInputMinVecIndex);
      float* input_min_vec_data = (float*)const_cast<void*>(
          static_cast<const void*>(input_min_vec.flat<float>().data()));
      const Tensor& input_max_vec = ctx->input(kInputMaxVecIndex);
      float* input_max_vec_data = (float*)const_cast<void*>(
          static_cast<const void*>(input_max_vec.flat<float>().data()));

      const Tensor& input_requested_min = ctx->input(this->kRequestMinIndex);
      const float input_requested_min_float =
          input_requested_min.flat<float>()(0);
      const Tensor& input_requested_max = ctx->input(this->kRequestMaxIndex);
      const float input_requested_max_float =
          input_requested_max.flat<float>()(0);

      size_t depth = input_min_vec.NumElements();
      OP_REQUIRES(
          ctx, input_min_vec.dim_size(0) == depth,
          errors::InvalidArgument("input_min has incorrect size, expected ",
                                  depth, " was ", input_min_vec.dim_size(0)));
      OP_REQUIRES(
          ctx, input_max_vec.dim_size(0) == depth,
          errors::InvalidArgument("input_max has incorrect size, expected ",
                                  depth, " was ", input_max_vec.dim_size(0)));

      if (out_type_ == DT_QINT8) DCHECK(input_requested_min_float < 0.0f);

      const float factor = (out_type_ == DT_QINT8) ? 127.0f : 255.0f;
      const float requested_min_max =
          std::max(std::abs(input_requested_min_float),
                   std::abs(input_requested_max_float));
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(kOutputTensorIndex,
                                               input.shape(), &output));

      std::vector<float> scales(depth);
      for (int i = 0; i < depth; ++i) {
        float min_max_from_vec = std::max(std::abs(input_min_vec_data[i]),
                                          std::abs(input_max_vec_data[i]));
        scales[i] = factor * (min_max_from_vec / requested_min_max /
                              static_cast<float>(1L << 31));
      }

      mkldnn::primitive_attr reorder_attr;
      reorder_attr.set_output_scales(2, scales);

      memory::dims dims_mkl_order =
          TFShapeToMklDnnDimsInNCHW(input.shape(), FORMAT_NHWC);
      memory::desc input_md = memory::desc(dims_mkl_order, MklDnnType<qint32>(),
                                           memory::format::nhwc);
      memory::desc output_md =
          (out_type_ == DT_QINT8)
              ? memory::desc(dims_mkl_order, MklDnnType<qint8>(),
                             memory::format::nhwc)
              : memory::desc(dims_mkl_order, MklDnnType<quint8>(),
                             memory::format::nhwc);

      memory::primitive_desc input_pd =
          memory::primitive_desc(input_md, cpu_engine_);
      memory::primitive_desc output_pd =
          memory::primitive_desc(output_md, cpu_engine_);

      void* input_buf =
          static_cast<void*>(const_cast<qint32*>(input.flat<qint32>().data()));
      void* output_buf;
      if (out_type_ == DT_QINT8) {
        output_buf = static_cast<void*>(
            const_cast<qint8*>(output->flat<qint8>().data()));
      } else {
        output_buf = static_cast<void*>(
            const_cast<quint8*>(output->flat<quint8>().data()));
      }

      std::unique_ptr<memory> input_mem_prim_(new memory(input_pd, input_buf));
      std::unique_ptr<memory> output_mem_prim_(
          new memory(output_pd, output_buf));

      mkldnn::reorder::primitive_desc reorder_pd =
          mkldnn::reorder::primitive_desc(input_pd, output_pd, reorder_attr);
      std::vector<mkldnn::primitive> net;
      net.push_back(
          mkldnn::reorder(reorder_pd, *input_mem_prim_, *output_mem_prim_));
      stream(stream::kind::eager).submit(net).wait();

      Tensor* output_min = nullptr;
      Tensor* output_max = nullptr;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(kOutputMinIndex, {}, &output_min));
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(kOutputMaxIndex, {}, &output_max));

      output_min->flat<float>()(0) = input_requested_min_float;
      output_max->flat<float>()(0) = input_requested_max_float;
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + std::string(e.message) + ", in file " +
                         std::string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          ctx, errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 private:
  const int kInputTensorIndex = 0;
  const int kInputMinVecIndex = 1;
  const int kInputMaxVecIndex = 2;
  const int kRequestMinIndex = 3;
  const int kRequestMaxIndex = 4;
  const int kOutputTensorIndex = 0;
  const int kOutputMinIndex = 1;
  const int kOutputMaxIndex = 2;
  DataType out_type_;
  engine cpu_engine_ = engine(engine::cpu, 0);
};

// Registration for out_type: qint8
REGISTER_KERNEL_BUILDER(Name("RequantizePerChannel")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint32>("T")
                            .TypeConstraint<qint8>("out_type"),
                        MklRequantizePerChannelOp<CPUDevice, qint8>);
// Registration for out_type: quint8
REGISTER_KERNEL_BUILDER(Name("RequantizePerChannel")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint32>("T")
                            .TypeConstraint<quint8>("out_type"),
                        MklRequantizePerChannelOp<CPUDevice, quint8>);

}  // namespace tensorflow
#endif  // INTEL_MKL
