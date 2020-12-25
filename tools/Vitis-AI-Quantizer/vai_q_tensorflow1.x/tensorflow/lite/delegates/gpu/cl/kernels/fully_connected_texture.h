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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_FULLY_CONNECTED_TEXTURE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_FULLY_CONNECTED_TEXTURE_H_

#include <vector>

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

class FullyConnectedTexture : public GPUOperation {
 public:
  FullyConnectedTexture() = default;
  Status AddToQueue(CLCommandQueue* queue) override;

  Status Compile(const CreationContext& creation_context) override;

  // Move only
  FullyConnectedTexture(FullyConnectedTexture&& kernel);
  FullyConnectedTexture& operator=(FullyConnectedTexture&& kernel);
  FullyConnectedTexture(const FullyConnectedTexture&) = delete;
  FullyConnectedTexture& operator=(const FullyConnectedTexture&) = delete;

 private:
  explicit FullyConnectedTexture(const OperationDef& definition);
  friend Status CreateFullyConnectedTexture(
      const CreationContext& creation_context, const OperationDef& definition,
      const FullyConnectedAttributes& attr, FullyConnectedTexture* result);

  template <DataType T>
  Status UploadWeights(const ::tflite::gpu::Tensor<OHWI, T>& weights,
                       CLContext* context);

  template <DataType T>
  void RearrangeWeightsFP16(const ::tflite::gpu::Tensor<OHWI, T>& weights,
                            absl::Span<half4> dst);
  template <DataType T>
  void RearrangeWeightsFP32(const ::tflite::gpu::Tensor<OHWI, T>& weights,
                            absl::Span<float4> dst);

  Texture2D weights_;
  LinearStorage biases_;
  CLKernel kernel_;
  int3 work_group_size_ = int3(0, 0, 0);
};

template <DataType T>
Status FullyConnectedTexture::UploadWeights(
    const ::tflite::gpu::Tensor<OHWI, T>& weights, CLContext* context) {
  const int src_depth = AlignByN(IntegralDivideRoundUp(weights.shape.i, 4), 4);
  const int dst_depth = IntegralDivideRoundUp(weights.shape.o, 4);

  if (definition_.GetDataType() == DataType::FLOAT32) {
    std::vector<float4> gpu_data(dst_depth * src_depth * 4);
    RearrangeWeightsFP32(weights, absl::MakeSpan(gpu_data));
    return CreateTexture2DRGBA(DataType::FLOAT32, dst_depth * 4, src_depth,
                               gpu_data.data(), context, &weights_);
  } else {
    std::vector<half4> gpu_data(dst_depth * src_depth * 4);
    RearrangeWeightsFP16(weights, absl::MakeSpan(gpu_data));
    return CreateTexture2DRGBA(DataType::FLOAT32, dst_depth, src_depth * 2,
                               gpu_data.data(), context, &weights_);
  }
}

template <DataType T>
void FullyConnectedTexture::RearrangeWeightsFP16(
    const ::tflite::gpu::Tensor<OHWI, T>& weights, absl::Span<half4> dst) {
  const int src_depth = AlignByN(IntegralDivideRoundUp(weights.shape.i, 4), 4);
  const int dst_depth = IntegralDivideRoundUp(weights.shape.o, 4);
  int counter = 0;

  for (int s = 0; s < src_depth; ++s) {
    for (int d = 0; d < dst_depth; ++d) {
      half4 filters[2];
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
          const int dst_ch = d * 4 + i;
          const int src_ch = s * 4 + j;
          if (dst_ch < weights.shape.o && src_ch < weights.shape.i) {
            const int f_index =
                weights.shape.LinearIndex({dst_ch, 0, 0, src_ch});
            filters[i][j] = weights.data[f_index];
          } else {
            filters[i][j] = 0.0;
          }
        }
      }
      dst[counter++] = filters[0];
      dst[counter++] = filters[1];
    }
    for (int d = 0; d < dst_depth; ++d) {
      half4 filters[2];
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
          const int dst_ch = d * 4 + 2 + i;
          const int src_ch = s * 4 + j;
          if (dst_ch < weights.shape.o && src_ch < weights.shape.i) {
            const int f_index =
                weights.shape.LinearIndex({dst_ch, 0, 0, src_ch});
            filters[i][j] = weights.data[f_index];
          } else {
            filters[i][j] = 0.0;
          }
        }
      }
      dst[counter++] = filters[0];
      dst[counter++] = filters[1];
    }
  }
}

template <DataType T>
void FullyConnectedTexture::RearrangeWeightsFP32(
    const ::tflite::gpu::Tensor<OHWI, T>& weights, absl::Span<float4> dst) {
  const int src_depth = AlignByN(IntegralDivideRoundUp(weights.shape.i, 4), 4);
  const int dst_depth = IntegralDivideRoundUp(weights.shape.o, 4);
  int counter = 0;

  for (int s = 0; s < src_depth; ++s) {
    for (int d = 0; d < dst_depth; ++d) {
      float4 filters[4];
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          const int dst_ch = d * 4 + i;
          const int src_ch = s * 4 + j;
          if (dst_ch < weights.shape.o && src_ch < weights.shape.i) {
            const int f_index =
                weights.shape.LinearIndex({dst_ch, 0, 0, src_ch});
            filters[i][j] = weights.data[f_index];
          } else {
            filters[i][j] = 0.0;
          }
        }
      }
      dst[counter++] = filters[0];
      dst[counter++] = filters[1];
      dst[counter++] = filters[2];
      dst[counter++] = filters[3];
    }
  }
}

Status CreateFullyConnectedTexture(const CreationContext& creation_context,
                                   const OperationDef& definition,
                                   const FullyConnectedAttributes& attr,
                                   FullyConnectedTexture* result);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_FULLY_CONNECTED_TEXTURE_H_
