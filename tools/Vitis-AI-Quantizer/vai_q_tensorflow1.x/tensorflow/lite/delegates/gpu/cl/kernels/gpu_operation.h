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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_GPU_OPERATION_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_GPU_OPERATION_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/tuning_parameters.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"
#include "tensorflow/lite/delegates/gpu/cl/program_cache.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

struct CreationContext {
  const CLDevice* device;
  CLContext* context;
  CLCommandQueue* queue;
  ProgramCache* cache;
};

struct OperationDef {
  CalculationsPrecision precision;
  std::vector<TensorDescriptor> src_tensors;
  std::vector<TensorDescriptor> dst_tensors;

  // returns FLOAT32 for F32 precision and FLOAT16 for F16 precision
  DataType GetDataType() const;
  // Primary means the first src tensor, because first tensor usually defines
  // the structure of kernel, all other resources(biases) types and etc.
  DataType GetPrimaryDataType() const;
  TensorStorageType GetPrimaryStorageType() const;
};

class ElementwiseOperation;

// GPUOperation represents some implementation of neural network operation on
// GPU. GPUOperation can contain ElementwiseOperation operations, in this case,
// ElementwiseOperation still hold necessary data and should be alive.
// When GPUOperation contains ElementwiseOperations, this GPUoperation replaces
// some sequence of operations Op + el_op0 + el_op1 + ...
// Because of this abilities of GPUOperation, usage scenario is next:
// Create instance of GPUOperation.
// Create all instances of ElementwiseOperations that we will(probably) attach
// to GPUOperation. Attach all ElementwiseOperations to GPUOperation. Call
// GPUOperation.Compile(). Don't call ElementwiseOperation.Compile() if it
// attached, it useless(and may be error)
class GPUOperation {
 public:
  GPUOperation() = default;
  explicit GPUOperation(const OperationDef& definition);
  virtual ~GPUOperation() = default;
  // Move only
  GPUOperation(GPUOperation&& operation);
  GPUOperation& operator=(GPUOperation&& operation);
  GPUOperation(const GPUOperation&) = delete;
  GPUOperation& operator=(const GPUOperation&) = delete;

  void AddOperation(ElementwiseOperation* operation);

  void SetSrc(Tensor* ptr, int index = 0);
  void SetDst(Tensor* ptr, int index = 0);

  virtual Status AddToQueue(CLCommandQueue* queue) { return OkStatus(); }
  virtual Status Tune(const TuningParameters& params) { return OkStatus(); }

  virtual Status Compile(const CreationContext& creation_context) {
    return OkStatus();
  }

  const OperationDef& GetDefinition() const { return definition_; }

 protected:
  // Defines operation calculation precision and format of src/dst tensors.
  OperationDef definition_;
  std::vector<Tensor*> src_;
  std::vector<Tensor*> dst_;
  std::vector<ElementwiseOperation*> linked_operations_;
};

// ElementwiseOperation can be fused(linked) to another operation.
// field linked_ indicate about this
// link_index_ used mostly for generating of correct names for
//   linked code variables
// link_index_ is number of operation in sequence of linked operations
// and should be unique in this sequence
// link_index_ = 0 is equivalent that operation not linked.
class ElementwiseOperation : public GPUOperation {
 public:
  ElementwiseOperation() {}
  explicit ElementwiseOperation(const OperationDef& definition)
      : GPUOperation(definition) {}

  virtual ~ElementwiseOperation() {}
  Status AddToQueue(CLCommandQueue* queue) override;
  Status Tune(const TuningParameters& params) override;

  Status Compile(const CreationContext& creation_context) override;

  // Move only
  ElementwiseOperation(ElementwiseOperation&& operation);
  ElementwiseOperation& operator=(ElementwiseOperation&& operation);
  ElementwiseOperation(const ElementwiseOperation&) = delete;
  ElementwiseOperation& operator=(const ElementwiseOperation&) = delete;

  // We need this function for resolving naming conflicts.
  // Unfortunately we don't know upfront(at creation time) will be the operation
  // linked or not. Operation should be created and SetLinkIndex(0) must be
  // called to initialize specific for this op linked info, and this is mean
  // that operation is not linked. But if we decided to link it, we need update
  // operation linked info and use names for kernel arguments according to this
  // index(this is responsibility of particular implementation of
  // ElementwiseOperation to generate right names).
  virtual void SetLinkIndex(int index) {}

  virtual std::string GetCoreCode(const std::string& src,
                                  const std::string& z_coord,
                                  const std::string& address) const = 0;
  virtual std::string GetArgsDeclaration() const { return ""; }
  virtual Status BindArguments(CLKernel* kernel) { return OkStatus(); }

 protected:
  Status BindArguments();
  int3 GetGridSize() const;
  CLKernel kernel_;
  int3 work_group_size_ = int3(8, 4, 1);
};

// Generates arguments declarations string for elementwise
// operations in linked_ops.
// Every ElementwiseOperation can generate arguments declarations.
std::string GetArgsDeclaration(
    const std::vector<ElementwiseOperation*>& linked_ops);

// Generates shader code for every elementwise operation in
// linked_ops.
// linked_ops - vector of operations pointers
// var_name - name of variable in shader code that we update/change
// z_coord - name of variable in shader code for currently processed Z -
//   coordinate in 3D grid (WHC/XYZ) for tensor, this coordinate is in
//   layer/slice(group of 4 channels) space not in channels.
// global_address - name of variable for coordinates in 3D grid (WHC/XYZ) for
//   tensor, different tensor layouts encode this address differently.
std::string PostProcess(const std::vector<ElementwiseOperation*>& linked_ops,
                        const std::string& var_name, const std::string& z_coord,
                        const std::string& global_address);

// Binds arguments to given kernel for elementwise operations in
// linked_ops.
// Every ElementwiseOperation can bind her arguments.
Status BindArgs(CLKernel* kernel,
                const std::vector<ElementwiseOperation*>& linked_ops);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_GPU_OPERATION_H_
