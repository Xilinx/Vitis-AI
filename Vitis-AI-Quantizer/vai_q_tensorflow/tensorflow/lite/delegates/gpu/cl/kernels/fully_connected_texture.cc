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

#include "tensorflow/lite/delegates/gpu/cl/kernels/fully_connected_texture.h"

#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

// We split vec vec dot (every thread do vec vec dot product in basic
// vec mat mult) on 4 parts to create more threads
// tid.y thread process every 4-th element in vec vec dot
// Good results for ~1024 x 1024 sizes, for other can be written more
// otimized shaders

std::string GetFullyConnectedKernelCode(
    const TensorDescriptor& src_descriptor,
    const TensorDescriptor& dst_descriptor, CalculationsPrecision precision,
    const std::vector<ElementwiseOperation*>& linked_operations,
    const int3& work_group_size) {
  TensorCodeGenerator src_tensor("src_data", "src_size", src_descriptor);
  TensorCodeGenerator dst_tensor("dst_data", "dst_size", dst_descriptor);

  std::string c = GetCommonDefines(precision);

  switch (precision) {
    case CalculationsPrecision::F32:
      c += "#define READ_IMAGE read_imagef\n";
      break;
    case CalculationsPrecision::F32_F16:
    case CalculationsPrecision::F16:
      c += "#define READ_IMAGE read_imageh\n";
      break;
  }

  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ) + ",\n";
  c += "    __read_only image2d_t filters,\n";
  c += "    __read_only image2d_t biases";
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 src_size,             \n";
  c += "    int4 dst_size,             \n";
  c += "    int src_depth_x4          \n";
  c += ") {\n";
  c += "  int gid = get_global_id(0);\n";
  c += "  int2 tid = (int2)(get_local_id(0), get_local_id(1));\n";
  c += "  ACCUM_FLT4 s = (ACCUM_FLT4)(0.0f);\n";
  c += "  uint c = tid.y;\n";       // vector coord for every thread
  c += "  uint c2 = tid.y * 2;\n";  // it should be * 4, so as we have FLT4
  // but we keep half8 in float4 so, we have * 2 y_coord for texture
  c += "  for (int i = 0; i < src_depth_x4; ++i, c += 4, c2 += 8) {\n";
  c += "    FLT4 v = " +
       src_tensor.Read3D("0", "0", "c", TextureAddressMode::DONT_CARE) + ";\n";
  if (precision != CalculationsPrecision::F32) {
    c += "   half8 m0 = as_half8(read_imagef(filters, smp_none, (int2)(gid, "
         "c2+0)));\n";
    c += "   half8 m1 = as_half8(read_imagef(filters, smp_none, (int2)(gid, "
         "c2+1)));\n";
    c += "   s.x += (v.x * m0.s0 + v.y * m0.s1 + v.z * m0.s2 + v.w * m0.s3);\n";
    c += "   s.y += (v.x * m0.s4 + v.y * m0.s5 + v.z * m0.s6 + v.w * m0.s7);\n";
    c += "   s.z += (v.x * m1.s0 + v.y * m1.s1 + v.z * m1.s2 + v.w * m1.s3);\n";
    c += "   s.w += (v.x * m1.s4 + v.y * m1.s5 + v.z * m1.s6 + v.w * m1.s7);\n";
  } else {
    c += "   float4 m0 = read_imagef(filters, smp_none, (int2)(gid * 4 + 0, "
         "c));\n";
    c += "   float4 m1 = read_imagef(filters, smp_none, (int2)(gid * 4 + 1, "
         "c));\n";
    c += "   float4 m2 = read_imagef(filters, smp_none, (int2)(gid * 4 + 2, "
         "c));\n";
    c += "   float4 m3 = read_imagef(filters, smp_none, (int2)(gid * 4 + 3, "
         "c));\n";
    c += "   s.x += (v.x * m0.s0 + v.y * m0.s1 + v.z * m0.s2 + v.w * m0.s3);\n";
    c += "   s.y += (v.x * m1.s0 + v.y * m1.s1 + v.z * m1.s2 + v.w * m1.s3);\n";
    c += "   s.z += (v.x * m2.s0 + v.y * m2.s1 + v.z * m2.s2 + v.w * m2.s3);\n";
    c += "   s.w += (v.x * m3.s0 + v.y * m3.s1 + v.z * m3.s2 + v.w * m3.s3);\n";
  }
  c += "  }\n";
  c += "  __local ACCUM_FLT4 temp[" + std::to_string(work_group_size.x) + "][" +
       std::to_string(work_group_size.y) + "];\n";
  c += "  temp[tid.x][tid.y] = s;\n";
  c += "  barrier(CLK_LOCAL_MEM_FENCE);\n";
  c += "  if (tid.y == 0 && gid < dst_size.w) {\n";
  c += "    s += temp[tid.x][1];\n";
  c += "    s += temp[tid.x][2];\n";
  c += "    s += temp[tid.x][3];\n";
  c += "    FLT4 r0 = TO_FLT4(s) + READ_IMAGE(biases, smp_none, (int2)(gid, "
       "0));\n";
  c += "  " + dst_tensor.GetAddress("dst_adr", "0", "0", "gid") + "\n";
  c += PostProcess(linked_operations, "r0", "gid", "dst_adr");
  c += "  " + dst_tensor.Write3D("r0", "dst_adr") + "\n";
  c += "  }\n";
  c += "}\n";

  return c;
}
}  // namespace

FullyConnectedTexture::FullyConnectedTexture(const OperationDef& definition)
    : GPUOperation(definition) {}

FullyConnectedTexture::FullyConnectedTexture(FullyConnectedTexture&& kernel)
    : GPUOperation(std::move(kernel)),
      weights_(std::move(kernel.weights_)),
      biases_(std::move(kernel.biases_)),
      kernel_(std::move(kernel.kernel_)),
      work_group_size_(kernel.work_group_size_) {}

FullyConnectedTexture& FullyConnectedTexture::operator=(
    FullyConnectedTexture&& kernel) {
  if (this != &kernel) {
    weights_ = std::move(kernel.weights_), biases_ = std::move(kernel.biases_),
    kernel_ = std::move(kernel.kernel_);
    std::swap(work_group_size_, kernel.work_group_size_);
    GPUOperation::operator=(std::move(kernel));
  }
  return *this;
}

Status FullyConnectedTexture::Compile(const CreationContext& creation_context) {
  int wg_width = 32;
  int wg_height = 4;
  int work_items;
  do {
    work_group_size_ = {wg_width, wg_height, 1};
    wg_width /= 2;
    const auto code = GetFullyConnectedKernelCode(
        definition_.src_tensors[0], definition_.dst_tensors[0],
        definition_.precision, linked_operations_, work_group_size_);
    RETURN_IF_ERROR(creation_context.cache->GetOrCreateCLKernel(
        code, "main_function", *creation_context.context,
        *creation_context.device, &kernel_));
    work_items = work_group_size_.x * work_group_size_.y * work_group_size_.z;
  } while (work_items > kernel_.GetMaxWorkGroupSize());
  return OkStatus();
}

Status FullyConnectedTexture::AddToQueue(CLCommandQueue* queue) {
  const int src_depth_x4 = IntegralDivideRoundUp(src_[0]->Depth(), 4);
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(weights_.GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(biases_.GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetSizeWithDepth()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetSizeWithDepth()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_depth_x4));

  return queue->DispatchImplicit(kernel_, {dst_[0]->Depth(), 1, 1},
                                 work_group_size_);
}

Status CreateFullyConnectedTexture(const CreationContext& creation_context,
                                   const OperationDef& definition,
                                   const FullyConnectedAttributes& attr,
                                   FullyConnectedTexture* result) {
  *result = FullyConnectedTexture(definition);
  RETURN_IF_ERROR(
      result->UploadWeights(attr.weights, creation_context.context));
  LinearStorageCreateInfo create_info;
  create_info.storage_type = LinearStorageType::TEXTURE_2D;
  create_info.data_type = definition.GetDataType();
  create_info.aligned_size = attr.weights.shape.o;
  RETURN_IF_ERROR(CreateLinearStorage(
      create_info, attr.bias, creation_context.context, &result->biases_));
  return OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
