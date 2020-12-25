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

#include "tensorflow/lite/delegates/gpu/cl/kernels/convolution_transposed_3x3_thin.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GenerateConvolutionTransposedCode(
    const TensorDescriptor& src_descriptor,
    const TensorDescriptor& dst_descriptor, CalculationsPrecision precision,
    const LinearStorage& biases, int src_depth, int dst_depth,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  TensorCodeGenerator src_tensor("src_data", "src_size", src_descriptor);
  TensorCodeGenerator dst_tensor("dst_data", "dst_size", dst_descriptor);

  std::string c = GetCommonDefines(precision);

  switch (precision) {
    case CalculationsPrecision::F32:
    case CalculationsPrecision::F16:
      c += "#define CONV(R, SRC, F, i) \\\n";
      c += "  R += SRC.x * F[i + 0]; \\\n";
      c += "  R += SRC.y * F[i + 1]; \\\n";
      c += "  R += SRC.z * F[i + 2]; \\\n";
      c += "  R += SRC.w * F[i + 3];   \n";
      break;
    case CalculationsPrecision::F32_F16:
      c += "#define CONV(R, SRC, F, i) \\\n";
      c += "  R += convert_float4(SRC.x * F[i + 0] + SRC.y * F[i + 1]";
      c += "+ SRC.z * F[i + 2] + SRC.w * F[i + 3]);\n";
      break;
  }

  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ) + ",\n";
  c += "    __constant FLT4* filters,  \n";
  c += biases.GetDeclaration();
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 src_size,             \n";
  c += "    int4 dst_size              \n";
  c += ") {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  if (X >= src_size.x || Y >= src_size.y) return;\n";
  for (int d = 0; d < dst_depth; ++d) {
    const std::string layer = std::to_string(d);
    c += "  ACCUM_FLT4 r" + layer + "[2][2];\n";
    c += "  r" + layer + "[0][0] = (ACCUM_FLT4)(0.0f);\n";
    c += "  r" + layer + "[0][1] = (ACCUM_FLT4)(0.0f);\n";
    c += "  r" + layer + "[1][0] = (ACCUM_FLT4)(0.0f);\n";
    c += "  r" + layer + "[1][1] = (ACCUM_FLT4)(0.0f);\n";
  }
  int filters_index = 0;
  for (int s = 0; s < src_depth; ++s) {
    const std::string z = std::to_string(s);
    c += "  {\n";
    if (src_descriptor.storage_type == TensorStorageType::BUFFER) {
      c += "  bool x_in = X + 1 < src_size.x;\n";
      c += "  bool y_in = Y + 1 < src_size.y;\n";
      c += "  FLT4 src0 = " + src_tensor.Read3D("X", "Y", z) + ";\n";
      c += "  FLT4 src1 = (FLT4)(0.0);\n";
      c += "  FLT4 src2 = (FLT4)(0.0);\n";
      c += "  FLT4 src3 = (FLT4)(0.0);\n";
      c += "  if (x_in) {\n";
      c += "    src1 = " + src_tensor.Read3D("X + 1", "Y", z) + ";\n";
      c += "  }\n";
      c += "  if (y_in) {\n";
      c += "    src2 = " + src_tensor.Read3D("X", "Y + 1", z) + ";\n";
      c += "  }\n";
      c += "  if (x_in && y_in) {\n";
      c += "    src3 = " + src_tensor.Read3D("X + 1", "Y + 1", z) + ";\n";
      c += "  }\n";
    } else {
      c += "  FLT4 src0 = " + src_tensor.Read3D("X", "Y", z) + ";\n";
      c += "  FLT4 src1 = " + src_tensor.Read3D("X + 1", "Y", z) + ";\n";
      c += "  FLT4 src2 = " + src_tensor.Read3D("X", "Y + 1", z) + ";\n";
      c += "  FLT4 src3 = " + src_tensor.Read3D("X + 1", "Y + 1", z) + ";\n";
    }
    for (int d = 0; d < dst_depth; ++d) {
      const std::string layer = std::to_string(d);
      const std::string f_offset = std::to_string(filters_index);
      filters_index++;
      c += "  {\n";
      c += "  __constant FLT4* L0 = filters + 36 * " + f_offset + ";\n";
      c += "  CONV(r" + layer + "[0][0], src0, L0, 0);\n";
      c += "  CONV(r" + layer + "[0][1], src0, L0, 4);\n";
      c += "  CONV(r" + layer + "[0][1], src1, L0, 8);\n";
      c += "  CONV(r" + layer + "[1][0], src0, L0, 12);\n";
      c += "  CONV(r" + layer + "[1][0], src2, L0, 16);\n";
      c += "  CONV(r" + layer + "[1][1], src0, L0, 20);\n";
      c += "  CONV(r" + layer + "[1][1], src1, L0, 24);\n";
      c += "  CONV(r" + layer + "[1][1], src2, L0, 28);\n";
      c += "  CONV(r" + layer + "[1][1], src3, L0, 32);\n";
      c += "  }\n";
    }
    c += "  }\n";
  }
  c += "  X *= 2;\n";
  c += "  Y *= 2;\n";
  for (int d = 0; d < dst_depth; ++d) {
    const std::string layer = std::to_string(d);
    c += "  {\n";
    c += "  FLT4 bias_val = " + biases.ReadLinearFLT4(layer) + ";\n";
    for (int y = 0; y < 2; ++y) {
      for (int x = 0; x < 2; ++x) {
        c += "  {\n";
        c += "    FLT4 result = TO_FLT4(r" + layer + "[" + std::to_string(y) +
             "][" + std::to_string(x) + "]) + bias_val;\n";
        c += "    " +
             dst_tensor.GetAddress("address", "X + " + std::to_string(x),
                                   "Y + " + std::to_string(y), layer) +
             "\n";
        c += PostProcess(linked_operations, "result", layer, "address");
        c += "    " + dst_tensor.Write3D("result", "address") + "\n";
        c += "  }\n";
      }
    }
    c += "  }\n";
  }
  c += "}\n";

  return c;
}
}  // namespace

ConvolutionTransposed3x3Thin::ConvolutionTransposed3x3Thin(
    const OperationDef& definition, const ConvolutionTransposedAttributes& attr)
    : GPUOperation(definition),
      src_channels_(attr.weights.shape.i),
      dst_channels_(attr.weights.shape.o) {}

ConvolutionTransposed3x3Thin::ConvolutionTransposed3x3Thin(
    ConvolutionTransposed3x3Thin&& operation)
    : GPUOperation(std::move(operation)),
      weights_(std::move(operation.weights_)),
      biases_(std::move(operation.biases_)),
      src_channels_(operation.src_channels_),
      dst_channels_(operation.dst_channels_),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

ConvolutionTransposed3x3Thin& ConvolutionTransposed3x3Thin::operator=(
    ConvolutionTransposed3x3Thin&& operation) {
  if (this != &operation) {
    weights_ = std::move(operation.weights_);
    biases_ = std::move(operation.biases_);
    std::swap(src_channels_, operation.src_channels_);
    std::swap(dst_channels_, operation.dst_channels_);
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

Status ConvolutionTransposed3x3Thin::Compile(
    const CreationContext& creation_context) {
  const auto code = GenerateConvolutionTransposedCode(
      definition_.src_tensors[0], definition_.dst_tensors[0],
      definition_.precision, biases_, IntegralDivideRoundUp(src_channels_, 4),
      IntegralDivideRoundUp(dst_channels_, 4), linked_operations_);

  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

Status ConvolutionTransposed3x3Thin::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(weights_.GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(biases_.GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetSizeWithDepth()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetSizeWithDepth()));
  return OkStatus();
}

int3 ConvolutionTransposed3x3Thin::GetGridSize() const {
  const int grid_x = src_[0]->Width();
  const int grid_y = src_[0]->Height();
  const int grid_z = 1;
  return int3(grid_x, grid_y, grid_z);
}

Status ConvolutionTransposed3x3Thin::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

Status ConvolutionTransposed3x3Thin::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

bool IsConvolutionTransposed3x3ThinSupported(
    const CLDevice& device, const ConvolutionTransposedAttributes& attr) {
  return device.IsAdreno() && attr.weights.shape.o <= 8 &&
         attr.weights.shape.w == 3 && attr.weights.shape.h == 3 &&
         attr.stride.w == 2 && attr.stride.h == 2 &&
         attr.padding.prepended.w == 1 && attr.padding.prepended.h == 1;
}

Status CreateConvolutionTransposed3x3Thin(
    const CreationContext& creation_context, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr,
    ConvolutionTransposed3x3Thin* result) {
  if (!IsConvolutionTransposed3x3ThinSupported(*creation_context.device,
                                               attr)) {
    return InvalidArgumentError(
        "ConvolutionTransposed3x3Thin doesn't support this attributes");
  }
  *result = ConvolutionTransposed3x3Thin(definition, attr);
  RETURN_IF_ERROR(
      result->UploadWeights(attr.weights, creation_context.context));
  LinearStorageCreateInfo create_info;
  create_info.storage_type =
      DeduceLinearStorageType(definition.GetPrimaryStorageType());
  create_info.data_type = definition.GetDataType();
  create_info.name = "biases";
  create_info.aligned_size = attr.weights.shape.o;
  RETURN_IF_ERROR(CreateLinearStorage(
      create_info, attr.bias, creation_context.context, &result->biases_));

  return OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
