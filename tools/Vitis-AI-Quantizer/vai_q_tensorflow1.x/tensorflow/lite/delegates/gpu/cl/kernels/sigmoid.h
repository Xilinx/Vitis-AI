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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_SIGMOID_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_SIGMOID_H_

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"

namespace tflite {
namespace gpu {
namespace cl {

class Sigmoid : public ElementwiseOperation {
 public:
  explicit Sigmoid(const OperationDef& definition)
      : ElementwiseOperation(definition) {}

  // Move only
  Sigmoid(Sigmoid&& operation);
  Sigmoid& operator=(Sigmoid&& operation);
  Sigmoid(const Sigmoid&) = delete;
  Sigmoid& operator=(const Sigmoid&) = delete;

  std::string GetCoreCode(const std::string& src, const std::string& z_coord,
                          const std::string& address) const override;
};

Sigmoid CreateSigmoid(const OperationDef& definition);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_SIGMOID_H_
