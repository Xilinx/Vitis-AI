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

#include "tensorflow/lite/delegates/gpu/cl/kernels/convolution_transposed_thin.h"

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
    int src_depth, int dst_channels, const int2& kernel_size,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  TensorCodeGenerator src_tensor("src_data", "src_size", src_descriptor);
  TensorCodeGenerator dst_tensor("dst_data", "dst_size", dst_descriptor);

  std::string c = GetCommonDefines(precision);
  const std::string channel_x = dst_channels == 1 ? "" : ".x";
  const std::vector<std::string> channel = {channel_x, ".y", ".z", ".w"};

  const std::string type_postfix =
      dst_channels == 1 ? "" : std::to_string(dst_channels);

  std::string accum_type;

  switch (precision) {
    case CalculationsPrecision::F32:
    case CalculationsPrecision::F32_F16:
      accum_type = "float" + type_postfix;
      break;
    case CalculationsPrecision::F16:
      accum_type = "half" + type_postfix;
      break;
  }

  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ) + ",\n";
  c += "    __constant FLT4* filters";
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 src_size,             \n";
  c += "    int4 dst_size,             \n";
  c += "    FLT4 bias_value            \n";
  c += ") {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  if (X >= src_size.x || Y >= src_size.y) return;\n";
  c += "  " + accum_type + " r[" + std::to_string(kernel_size.y) + "][" +
       std::to_string(kernel_size.x) + "];\n";
  c += "  {\n";
  c += "  FLT4 src = " + src_tensor.Read3D("X", "Y", "0") + ";\n";
  int index = 0;
  for (int y = 0; y < kernel_size.y; ++y) {
    for (int x = 0; x < kernel_size.x; ++x) {
      std::string r_s =
          "  r[" + std::to_string(y) + "][" + std::to_string(x) + "]";
      for (int d = 0; d < dst_channels; ++d) {
        c += r_s + channel[d] + " = TO_ACCUM_FLT(dot(src, filters[" +
             std::to_string(index) + "]));\n";
        index++;
      }
    }
  }
  c += "  }\n";
  for (int i = 1; i < src_depth; ++i) {
    if (precision != CalculationsPrecision::F32_F16) {
      c += "  if (X < src_size.x + " + std::to_string(i + 1) + ") {\n";
    } else {
      c += "  {\n";
    }
    c += "  FLT4 src = " + src_tensor.Read3D("X", "Y", std::to_string(i)) +
         ";\n";
    for (int y = 0; y < kernel_size.y; ++y) {
      for (int x = 0; x < kernel_size.x; ++x) {
        std::string r_s =
            "  r[" + std::to_string(y) + "][" + std::to_string(x) + "]";
        for (int d = 0; d < dst_channels; ++d) {
          c += r_s + channel[d] + " += TO_ACCUM_FLT(dot(src, filters[" +
               std::to_string(index) + "]));\n";
          index++;
        }
      }
    }
    c += "  }\n";
  }
  c += "  X *= " + std::to_string(kernel_size.x) + ";\n";
  c += "  Y *= " + std::to_string(kernel_size.x) + ";\n";
  for (int y = 0; y < kernel_size.y; ++y) {
    for (int x = 0; x < kernel_size.x; ++x) {
      if (precision != CalculationsPrecision::F32_F16) {
        c += "  if (X + " + std::to_string(x) + " < dst_size.x && ";
        c += "Y + " + std::to_string(y) + " < dst_size.y) {\n";
      } else {
        c += "  {\n";
      }
      c += "    FLT4 result = bias_value;\n";
      for (int d = 0; d < dst_channels; ++d) {
        c += "    result" + channel[d] + " += r[" + std::to_string(y) + "][" +
             std::to_string(x) + "]" + channel[d] + ";\n";
      }
      c += "    " +
           dst_tensor.GetAddress("address", "X + " + std::to_string(x),
                                 "Y + " + std::to_string(y), "0") +
           "\n";
      c += PostProcess(linked_operations, "result", "0", "address");
      c += "    " + dst_tensor.Write3D("result", "address") + "\n";
      c += "  }\n";
    }
  }
  c += "}\n";

  return c;
}
}  // namespace

ConvolutionTransposedThin::ConvolutionTransposedThin(
    const OperationDef& definition, const ConvolutionTransposedAttributes& attr)
    : GPUOperation(definition),
      kernel_size_(attr.weights.shape.w, attr.weights.shape.h),
      src_channels_(attr.weights.shape.i),
      dst_channels_(attr.weights.shape.o) {
  float4 bias_value(0.0f);
  for (int i = 0; i < attr.weights.shape.o; ++i) {
    bias_value[i] = attr.bias.data[i];
  }
  bias_value_ = FLT4(definition_.precision, bias_value);
}

ConvolutionTransposedThin::ConvolutionTransposedThin(
    ConvolutionTransposedThin&& operation)
    : GPUOperation(std::move(operation)),
      weights_buf_(std::move(operation.weights_buf_)),
      bias_value_(std::move(operation.bias_value_)),
      kernel_size_(operation.kernel_size_),
      src_channels_(operation.src_channels_),
      dst_channels_(operation.dst_channels_),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

ConvolutionTransposedThin& ConvolutionTransposedThin::operator=(
    ConvolutionTransposedThin&& operation) {
  if (this != &operation) {
    weights_buf_ = std::move(operation.weights_buf_);
    bias_value_ = std::move(operation.bias_value_);
    std::swap(kernel_size_, operation.kernel_size_);
    std::swap(src_channels_, operation.src_channels_);
    std::swap(dst_channels_, operation.dst_channels_);
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

Status ConvolutionTransposedThin::Compile(
    const CreationContext& creation_context) {
  const auto code = GenerateConvolutionTransposedCode(
      definition_.src_tensors[0], definition_.dst_tensors[0],
      definition_.precision, IntegralDivideRoundUp(src_channels_, 4),
      dst_channels_, kernel_size_, linked_operations_);

  std::vector<CompilerOptions> options;
  if (definition_.precision == CalculationsPrecision::F16 &&
      creation_context.device->IsAdreno3xx()) {
    options.push_back(CompilerOptions::ADRENO_FULL_SIMD_LINE);
  }

  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

Status ConvolutionTransposedThin::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(weights_buf_.GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetSizeWithDepth()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetSizeWithDepth()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(bias_value_));
  return OkStatus();
}

int3 ConvolutionTransposedThin::GetGridSize() const {
  const int grid_x = src_[0]->Width();
  const int grid_y = src_[0]->Height();
  const int grid_z = 1;
  return int3(grid_x, grid_y, grid_z);
}

Status ConvolutionTransposedThin::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

Status ConvolutionTransposedThin::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

bool IsConvolutionTransposedThinSupported(
    const CLDevice& device, const ConvolutionTransposedAttributes& attr) {
  return device.IsAdreno() && attr.weights.shape.o <= 4 &&
         attr.weights.shape.w == attr.stride.w &&
         attr.weights.shape.h == attr.stride.h &&
         attr.padding.prepended.w == 0 && attr.padding.prepended.h == 0;
}

Status CreateConvolutionTransposedThin(
    const CreationContext& creation_context, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr,
    ConvolutionTransposedThin* result) {
  if (!IsConvolutionTransposedThinSupported(*creation_context.device, attr)) {
    return InvalidArgumentError(
        "ConvolutionTransposedThin doesn't support this attributes");
  }
  *result = ConvolutionTransposedThin(definition, attr);
  RETURN_IF_ERROR(
      result->UploadWeights(attr.weights, creation_context.context));
  return OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
