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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_H_

#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/cl/linear_storage.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
#include "tensorflow/lite/delegates/gpu/cl/texture2d.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

class ConvolutionTransposed : public GPUOperation {
 public:
  ConvolutionTransposed() = default;
  Status AddToQueue(CLCommandQueue* queue) override;
  Status Tune(const TuningParameters& params) override;

  Status Compile(const CreationContext& creation_context) override;

  // Move only
  ConvolutionTransposed(ConvolutionTransposed&& kernel);
  ConvolutionTransposed& operator=(ConvolutionTransposed&& kernel);
  ConvolutionTransposed(const ConvolutionTransposed&) = delete;
  ConvolutionTransposed& operator=(const ConvolutionTransposed&) = delete;

 private:
  friend Status CreateConvolutionTransposed(
      const CreationContext& creation_context, const OperationDef& definition,
      const ConvolutionTransposedAttributes& attr,
      ConvolutionTransposed* result);
  explicit ConvolutionTransposed(const OperationDef& definition,
                                 const ConvolutionTransposedAttributes& attr);
  template <DataType T>
  Status UploadWeights(const ::tflite::gpu::Tensor<OHWI, T>& weights,
                       CLContext* context);

  template <DataType S, typename T>
  void RearrangeWeightsData(const ::tflite::gpu::Tensor<OHWI, S>& weights,
                            absl::Span<T> dst);

  Status BindArguments();
  int3 GetGridSize() const;

  LinearStorage biases_;

  Texture2D weights_tex2d_;
  Buffer weights_buf_;
  cl_mem weights_;

  int2 kernel_size_;
  int2 stride_;
  int2 padding_;
  int2 kernel_offset_;
  int2 inner_size_;
  int src_channels_;
  int dst_channels_;

  CLKernel kernel_;
  int3 work_group_size_ = int3(16, 8, 1);
};

template <DataType T>
Status ConvolutionTransposed::UploadWeights(
    const ::tflite::gpu::Tensor<OHWI, T>& weights, CLContext* context) {
  const int dst_depth = IntegralDivideRoundUp(dst_channels_, 4);
  const int src_depth = IntegralDivideRoundUp(src_channels_, 4);
  const int kernel_x = kernel_size_.x;
  const int kernel_y = kernel_size_.y;

  const int elements_count = kernel_x * kernel_y * src_depth * dst_depth * 4;
  bool is_buffer_storage =
      definition_.GetPrimaryStorageType() == TensorStorageType::BUFFER;

  const int float4_size =
      definition_.precision == CalculationsPrecision::F32 ? 16 : 8;

  if (definition_.GetDataType() == DataType::FLOAT32) {
    std::vector<float4> gpu_data(elements_count);
    RearrangeWeightsData(weights, absl::MakeSpan(gpu_data));
    if (is_buffer_storage) {
      RETURN_IF_ERROR(CreateReadOnlyBuffer(float4_size * elements_count,
                                           gpu_data.data(), context,
                                           &weights_buf_));
    } else {
      RETURN_IF_ERROR(CreateTexture2DRGBA(
          definition_.GetDataType(), src_depth * kernel_x * kernel_y * 4,
          dst_depth, gpu_data.data(), context, &weights_tex2d_));
    }
  } else {
    std::vector<half4> gpu_data(elements_count);
    RearrangeWeightsData(weights, absl::MakeSpan(gpu_data));
    if (is_buffer_storage) {
      RETURN_IF_ERROR(CreateReadOnlyBuffer(float4_size * elements_count,
                                           gpu_data.data(), context,
                                           &weights_buf_));
    } else {
      RETURN_IF_ERROR(CreateTexture2DRGBA(
          definition_.GetDataType(), src_depth * kernel_x * kernel_y * 4,
          dst_depth, gpu_data.data(), context, &weights_tex2d_));
    }
  }

  if (is_buffer_storage) {
    weights_ = weights_buf_.GetMemoryPtr();
  } else {
    weights_ = weights_tex2d_.GetMemoryPtr();
  }

  return OkStatus();
}

template <DataType S, typename T>
void ConvolutionTransposed::RearrangeWeightsData(
    const ::tflite::gpu::Tensor<OHWI, S>& weights, absl::Span<T> dst) {
  const int dst_depth = IntegralDivideRoundUp(dst_channels_, 4);
  const int src_depth = IntegralDivideRoundUp(src_channels_, 4);
  const int kernel_x = kernel_size_.x;
  const int kernel_y = kernel_size_.y;

  int counter = 0;
  for (int d = 0; d < dst_depth; ++d) {
    for (int y = 0; y < kernel_y; ++y) {
      for (int x = 0; x < kernel_x; ++x) {
        for (int s = 0; s < src_depth; ++s) {
          T filters[4];
          for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) {
              const int s_ch = s * 4 + j;
              const int d_ch = d * 4 + i;
              if (s_ch < src_channels_ && d_ch < dst_channels_) {
                const int f_index =
                    weights.shape.LinearIndex({d_ch, y, x, s_ch});
                filters[i][j] = weights.data[f_index];
              } else {
                filters[i][j] = 0.0f;
              }
            }
          }
          T filters_new[4];
          for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
              filters_new[i][j] = filters[j][i];
            }
          }
          dst[counter++] = filters_new[0];
          dst[counter++] = filters_new[1];
          dst[counter++] = filters_new[2];
          dst[counter++] = filters_new[3];
        }
      }
    }
  }
}

Status CreateConvolutionTransposed(const CreationContext& creation_context,
                                   const OperationDef& definition,
                                   const ConvolutionTransposedAttributes& attr,
                                   ConvolutionTransposed* result);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVOLUTION_TRANSPOSED_H_
