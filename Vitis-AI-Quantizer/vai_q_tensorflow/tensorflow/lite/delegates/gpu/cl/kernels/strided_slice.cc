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

#include "tensorflow/lite/delegates/gpu/cl/kernels/strided_slice.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetStridedSliceCode(
    const TensorDescriptor& src_descriptor,
    const TensorDescriptor& dst_descriptor, CalculationsPrecision precision,
    bool alignedx4,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  TensorCodeGenerator src_tensor("src_data", "src_size", src_descriptor);
  TensorCodeGenerator dst_tensor("dst_data", "dst_size", dst_descriptor);

  std::string c = GetCommonDefines(precision);
  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ);
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 offset,            \n";
  c += "    int4 stride,            \n";
  c += "    int4 src_size,             \n";
  c += "    int4 dst_size              \n";
  c += ") {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  int s_x = X * stride.x + offset.x;\n";
  c += "  int s_y = Y * stride.y + offset.y;\n";
  if (alignedx4) {
    c += "  int s_z = Z + offset.z;\n";
    c += "  FLT4 result = " +
         src_tensor.Read3D("s_x", "s_y", "s_z", TextureAddressMode::DONT_CARE) +
         ";\n";
  } else {
    c += "  FLT4 result;\n";
    const std::string postfixes[] = {"x", "y", "z", "w"};
    for (int i = 0; i < 4; ++i) {
      c += "  {\n";
      const std::string channel = "(Z * 4 + " + std::to_string(i) + ")";
      c += "    int s_ch = " + channel + " * stride.z + offset.z;\n";
      c += "    int s_z = s_ch >> 2;\n";
      c += "    int s_z_rem = s_ch & 3;\n";
      c += "    FLT4 t = " +
           src_tensor.Read3D("s_x", "s_y", "s_z",
                             TextureAddressMode::DONT_CARE) +
           ";\n";
      c += "    FLT t_ar[4] = {t.x, t.y, t.z, t.w};\n";
      c += "    result." + postfixes[i] + " = t_ar[s_z_rem];\n";
      c += "  }\n";
    }
  }
  c += "  " + dst_tensor.GetAddress("dst_adr", "X", "Y", "Z");
  c += PostProcess(linked_operations, "result", "Z", "dst_adr");
  c += "  " + dst_tensor.Write3D("result", "dst_adr");
  c += "}\n";
  return c;
}

bool Is4Alighed(const SliceAttributes& attr) {
  return attr.strides.c == 1 && attr.starts.c % 4 == 0;
}

int3 GetOffset(const SliceAttributes& attr, int src_width, int src_height,
               int src_channels) {
  int3 offset;
  if (attr.strides.w > 0) {
    offset.x = attr.starts.w;
  } else {
    if (attr.ends.w > 0) {
      offset.x = attr.ends.w;
    } else {
      offset.x = src_width + attr.ends.w;
    }
  }
  if (attr.strides.h > 0) {
    offset.y = attr.starts.h;
  } else {
    if (attr.ends.h > 0) {
      offset.y = attr.ends.h;
    } else {
      offset.y = src_height + attr.ends.h;
    }
  }
  if (attr.strides.c > 0) {
    offset.z = attr.starts.c;
  } else {
    if (attr.ends.c > 0) {
      offset.z = attr.ends.c;
    } else {
      offset.z = src_channels + attr.ends.c;
    }
  }
  if (Is4Alighed(attr)) {
    offset.z /= 4;
  }
  return offset;
}

}  // namespace

StridedSlice::StridedSlice(const OperationDef& definition,
                           const SliceAttributes& attr)
    : GPUOperation(definition), attributes_(attr), work_group_size_(8, 4, 1) {}

StridedSlice::StridedSlice(StridedSlice&& operation)
    : GPUOperation(std::move(operation)),
      attributes_(operation.attributes_),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

StridedSlice& StridedSlice::operator=(StridedSlice&& operation) {
  if (this != &operation) {
    attributes_ = operation.attributes_;
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

Status StridedSlice::Compile(const CreationContext& creation_context) {
  const auto code = GetStridedSliceCode(
      definition_.src_tensors[0], definition_.dst_tensors[0],
      definition_.precision, Is4Alighed(attributes_), linked_operations_);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

Status StridedSlice::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtr()));
  int3 offset = GetOffset(attributes_, src_[0]->Width(), src_[0]->Height(),
                          src_[0]->Channels());
  RETURN_IF_ERROR(kernel_.SetBytesAuto(int4(offset.x, offset.y, offset.z, 1)));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(int4(
      attributes_.strides.w, attributes_.strides.h, attributes_.strides.c, 1)));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetSizeWithDepth()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetSizeWithDepth()));
  return OkStatus();
}

int3 StridedSlice::GetGridSize() const {
  const int grid_x = dst_[0]->Width();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Depth();
  return int3(grid_x, grid_y, grid_z);
}

Status StridedSlice::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

Status StridedSlice::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

StridedSlice CreateStridedSlice(const OperationDef& definition,
                                const SliceAttributes& attr) {
  return StridedSlice(definition, attr);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
