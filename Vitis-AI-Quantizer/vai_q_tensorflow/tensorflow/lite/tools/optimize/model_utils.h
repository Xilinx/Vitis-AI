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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_MODEL_UTILS_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_MODEL_UTILS_H_

#include "absl/memory/memory.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace optimize {
namespace utils {

// Creates a Dequantize OperatorT object.
void MakeDequantizeOperator(ModelT* model, std::unique_ptr<OperatorT>* op,
                            int32_t input, int32_t output);

// Creates a Quantize OperatorT object.
void MakeQuantizeOperator(ModelT* model, std::unique_ptr<OperatorT>* op,
                          int32_t input, int32_t output);

// Create a new TensorT object without quantization parameters.
void MakeTensor(const string& name, const std::vector<int32_t>& shape,
                const TensorType& type, std::unique_ptr<TensorT>* tensor);

// Create a new TensorT object with quantization parameters.
void MakeTensorWithQuantParam(const string& name,
                              const std::vector<int32_t>& shape,
                              const TensorType& type, float scale,
                              int64_t zero_point,
                              std::unique_ptr<TensorT>* tensor);

// Checks if the tensor has scale and zero point populated.
bool QuantizationParametersExist(const TensorT* tensor);

bool HasBuffer(const ModelT* model, const SubGraphT* subgraph,
               int tensor_index);

bool IsQuantized(const SubGraphT* subgraph, int tensor_index);

bool HasMinMax(const TensorT* tensor);

// Set version of OperatorCode. The version will only be applied for operations
// that have been quantized.
void SetOperatorCodeVersion(ModelT* model);

}  // namespace utils
}  // namespace optimize
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_MODEL_UTILS_H_
