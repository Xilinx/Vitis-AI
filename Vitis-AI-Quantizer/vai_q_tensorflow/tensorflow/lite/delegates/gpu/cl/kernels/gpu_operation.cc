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

#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/access_type.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetElementWiseCode(
    const TensorDescriptor& src_descriptor,
    const TensorDescriptor& dst_descriptor, CalculationsPrecision precision,
    const ElementwiseOperation& op,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  TensorCodeGenerator src_tensor("src_data", "dst_size", src_descriptor);
  TensorCodeGenerator dst_tensor("dst_data", "dst_size", dst_descriptor);

  std::string c = GetCommonDefines(precision);

  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ);
  c += op.GetArgsDeclaration();
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 dst_size\n";
  c += ") {\n";
  c += "  int X = get_global_id(0);\n";
  c += "  int Y = get_global_id(1);\n";
  c += "  int Z = get_global_id(2);\n";
  c += "  if (X >= dst_size.x || Y >= dst_size.y) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  " + src_tensor.GetAddress("address", "X", "Y", "Z") + "\n";
  c += "  FLT4 src = " + src_tensor.Read3D("address") + ";\n";
  c += "  " + op.GetCoreCode("src", "Z", "address");
  c += PostProcess(linked_operations, "src", "Z", "address");
  c += "  " + dst_tensor.Write3D("src", "address") + "\n";
  c += "} \n";
  return c;
}

}  // namespace

DataType OperationDef::GetDataType() const {
  return DeduceDataTypeFromPrecision(precision);
}

DataType OperationDef::GetPrimaryDataType() const {
  return src_tensors[0].data_type;
}
TensorStorageType OperationDef::GetPrimaryStorageType() const {
  return src_tensors[0].storage_type;
}

GPUOperation::GPUOperation(const OperationDef& definition)
    : definition_(definition) {}

void GPUOperation::SetSrc(Tensor* ptr, int index) {
  if (index >= src_.size()) {
    src_.resize(index + 1, nullptr);
  }
  src_[index] = ptr;
}

void GPUOperation::SetDst(Tensor* ptr, int index) {
  if (index >= dst_.size()) {
    dst_.resize(index + 1, nullptr);
  }
  dst_[index] = ptr;
}

GPUOperation::GPUOperation(GPUOperation&& operation)
    : definition_(std::move(operation.definition_)),
      src_(std::move(operation.src_)),
      dst_(std::move(operation.dst_)),
      linked_operations_(std::move(operation.linked_operations_)) {}

GPUOperation& GPUOperation::operator=(GPUOperation&& operation) {
  if (this != &operation) {
    definition_ = std::move(operation.definition_);
    src_ = std::move(operation.src_);
    dst_ = std::move(operation.dst_);
    linked_operations_ = std::move(operation.linked_operations_);
  }
  return *this;
}

void GPUOperation::AddOperation(ElementwiseOperation* operation) {
  linked_operations_.push_back(operation);
  operation->SetLinkIndex(linked_operations_.size());
}

ElementwiseOperation::ElementwiseOperation(ElementwiseOperation&& operation)
    : GPUOperation(std::move(operation)),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

ElementwiseOperation& ElementwiseOperation::operator=(
    ElementwiseOperation&& operation) {
  if (this != &operation) {
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

Status ElementwiseOperation::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(BindArguments(&kernel_));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetSizeWithDepth()));
  return OkStatus();
}

int3 ElementwiseOperation::GetGridSize() const {
  const int grid_x = dst_[0]->Width();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Depth();
  return int3(grid_x, grid_y, grid_z);
}

Status ElementwiseOperation::Compile(const CreationContext& creation_context) {
  const auto code =
      GetElementWiseCode(definition_.src_tensors[0], definition_.dst_tensors[0],
                         definition_.precision, *this, linked_operations_);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

Status ElementwiseOperation::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

Status ElementwiseOperation::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

std::string GetArgsDeclaration(
    const std::vector<ElementwiseOperation*>& linked_ops) {
  std::string code;
  for (auto linked_op : linked_ops) {
    code += linked_op->GetArgsDeclaration();
  }
  code += ",\n";

  return code;
}

std::string PostProcess(const std::vector<ElementwiseOperation*>& linked_ops,
                        const std::string& var_name, const std::string& z_coord,
                        const std::string& global_address) {
  std::string code;
  for (auto linked_op : linked_ops) {
    code += linked_op->GetCoreCode(var_name, z_coord, global_address);
  }
  return code;
}

Status BindArgs(CLKernel* kernel,
                const std::vector<ElementwiseOperation*>& linked_ops) {
  for (auto linked_op : linked_ops) {
    RETURN_IF_ERROR(linked_op->BindArguments(kernel));
  }
  return OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
