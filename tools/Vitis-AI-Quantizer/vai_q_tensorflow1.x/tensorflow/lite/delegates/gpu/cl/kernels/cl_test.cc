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

#include "tensorflow/lite/delegates/gpu/cl/kernels/cl_test.h"

#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

Status ExecuteGPUOperation(const std::vector<TensorFloat32>& src_cpu,
                           const CreationContext& creation_context,
                           GPUOperation* operation,
                           const std::vector<BHWC>& dst_sizes,
                           const std::vector<TensorFloat32*>& dst_cpu) {
  const OperationDef& op_def = operation->GetDefinition();
  std::vector<Tensor> src(src_cpu.size());
  for (int i = 0; i < src_cpu.size(); ++i) {
    auto src_shape = src_cpu[i].shape;
    RETURN_IF_ERROR(CreateTensor(
        *creation_context.context, *creation_context.device, src_shape.w,
        src_shape.h, src_shape.c, op_def.src_tensors[0].data_type,
        op_def.src_tensors[0].storage_type, &src[i]));
    RETURN_IF_ERROR(src[i].WriteData(creation_context.queue, src_cpu[i]));
    operation->SetSrc(&src[i], i);
  }

  std::vector<Tensor> dst(dst_cpu.size());
  for (int i = 0; i < dst_cpu.size(); ++i) {
    auto dst_shape = dst_sizes[i];
    RETURN_IF_ERROR(CreateTensor(
        *creation_context.context, *creation_context.device, dst_shape.w,
        dst_shape.h, dst_shape.c, op_def.dst_tensors[0].data_type,
        op_def.dst_tensors[0].storage_type, &dst[i]));

    operation->SetDst(&dst[i], i);
  }

  RETURN_IF_ERROR(operation->Compile(creation_context));
  RETURN_IF_ERROR(operation->AddToQueue(creation_context.queue));
  RETURN_IF_ERROR(creation_context.queue->WaitForCompletion());

  for (int i = 0; i < dst_cpu.size(); ++i) {
    dst_cpu[i]->shape = dst_sizes[i];
    dst_cpu[i]->data = std::vector<float>(dst_sizes[i].DimensionsProduct(), 0);
    RETURN_IF_ERROR(dst[i].ReadData(creation_context.queue, dst_cpu[i]));
  }
  return OkStatus();
}

Status ExecuteGPUOperation(const std::vector<TensorFloat32>& src_cpu,
                           const CreationContext& creation_context,
                           GPUOperation* operation, const BHWC& dst_size,
                           TensorFloat32* result) {
  return ExecuteGPUOperation(
      std::vector<TensorFloat32>{src_cpu}, creation_context, operation,
      std::vector<BHWC>{dst_size}, std::vector<TensorFloat32*>{result});
}

Status ExecuteGPUOperation(const TensorFloat32& src_cpu,
                           const CreationContext& creation_context,
                           GPUOperation* operation, const BHWC& dst_size,
                           TensorFloat32* result) {
  return ExecuteGPUOperation(std::vector<TensorFloat32>{src_cpu},
                             creation_context, operation, dst_size, result);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
