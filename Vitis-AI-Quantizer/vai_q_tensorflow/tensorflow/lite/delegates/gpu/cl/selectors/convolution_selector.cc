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

#include "tensorflow/lite/delegates/gpu/cl/selectors/convolution_selector.h"

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_buffer_1x1.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_constants.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_texture.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

Status SelectConvolutionTextureArray(const Convolution2DAttributes& attr,
                                     const BHWC& dst_shape,
                                     const CreationContext& creation_context,
                                     const OperationDef& op_def,
                                     ModelHints hints,
                                     std::unique_ptr<GPUOperation>* ptr) {
  if (IsConvConstantsSupported(*creation_context.device, op_def, attr)) {
    ConvConstants conv;
    RETURN_IF_ERROR(CreateConvConstants(creation_context, op_def, attr, &conv));
    *ptr = absl::make_unique<ConvConstants>(std::move(conv));
  } else {
    ConvTexture conv;
    RETURN_IF_ERROR(CreateConvTexture(creation_context, op_def, attr, &conv));
    *ptr = absl::make_unique<ConvTexture>(std::move(conv));
  }

  return OkStatus();
}

Status SelectConvolutionTexture2D(const Convolution2DAttributes& attr,
                                  const CreationContext& creation_context,
                                  const OperationDef& op_def,
                                  std::unique_ptr<GPUOperation>* ptr) {
  if (IsConvConstantsSupported(*creation_context.device, op_def, attr)) {
    ConvConstants conv;
    RETURN_IF_ERROR(CreateConvConstants(creation_context, op_def, attr, &conv));
    *ptr = absl::make_unique<ConvConstants>(std::move(conv));
  } else {
    ConvTexture conv;
    RETURN_IF_ERROR(CreateConvTexture(creation_context, op_def, attr, &conv));
    *ptr = absl::make_unique<ConvTexture>(std::move(conv));
  }
  return OkStatus();
}

Status SelectConvolutionBuffer(const Convolution2DAttributes& attr,
                               const CreationContext& creation_context,
                               const OperationDef& op_def,
                               std::unique_ptr<GPUOperation>* ptr) {
  if (IsConvBuffer1x1Supported(op_def, attr)) {
    ConvBuffer1x1 conv;
    RETURN_IF_ERROR(CreateConvBuffer1x1(creation_context, op_def, attr, &conv));
    *ptr = absl::make_unique<ConvBuffer1x1>(std::move(conv));
  } else {
    ConvBuffer conv;
    RETURN_IF_ERROR(CreateConvBuffer(creation_context, op_def, attr, &conv));
    *ptr = absl::make_unique<ConvBuffer>(std::move(conv));
  }
  return OkStatus();
}
}  // namespace

Status SelectConvolution(const Convolution2DAttributes& attr,
                         const BHWC& dst_shape,
                         const CreationContext& creation_context,
                         const OperationDef& op_def, ModelHints hints,
                         std::unique_ptr<GPUOperation>* ptr) {
  switch (op_def.GetPrimaryStorageType()) {
    case TensorStorageType::TEXTURE_ARRAY:
      return SelectConvolutionTextureArray(attr, dst_shape, creation_context,
                                           op_def, hints, ptr);
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return SelectConvolutionTexture2D(attr, creation_context, op_def, ptr);
    case TensorStorageType::BUFFER:
      return SelectConvolutionBuffer(attr, creation_context, op_def, ptr);
    default:
      return InternalError("Unknown storage type.");
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
