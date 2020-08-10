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

#include "tensorflow/lite/delegates/gpu/cl/kernels/sigmoid.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"

namespace tflite {
namespace gpu {
namespace cl {

Sigmoid::Sigmoid(Sigmoid&& operation)
    : ElementwiseOperation(std::move(operation)) {}

Sigmoid& Sigmoid::operator=(Sigmoid&& operation) {
  if (this != &operation) {
    ElementwiseOperation::operator=(std::move(operation));
  }
  return *this;
}

std::string Sigmoid::GetCoreCode(const std::string& src,
                                 const std::string& z_coord,
                                 const std::string& address) const {
  if (definition_.precision != CalculationsPrecision::F32) {
    return absl::StrCat(
        src, ".x = convert_half(native_recip(1.0f + native_exp(convert_float(-",
        src, ".x))));\n", "  ", src,
        ".y = convert_half(native_recip(1.0f + native_exp(convert_float(-", src,
        ".y))));\n", "  ", src,
        ".z = convert_half(native_recip(1.0f + native_exp(convert_float(-", src,
        ".z))));\n", "  ", src,
        ".w = convert_half(native_recip(1.0f + native_exp(convert_float(-", src,
        ".w))));\n");
  } else {
    return absl::StrCat(src, " = (FLT4)(1.0f) / ((FLT4)(1.0f) + exp(-(", src,
                        ")));\n");
  }
}

Sigmoid CreateSigmoid(const OperationDef& definition) {
  Sigmoid operation(definition);
  operation.SetLinkIndex(0);
  return operation;
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
