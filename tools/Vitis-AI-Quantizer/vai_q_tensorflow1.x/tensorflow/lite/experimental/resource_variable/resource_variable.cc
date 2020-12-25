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

#include "tensorflow/lite/experimental/resource_variable/resource_variable.h"

#include <cstdlib>
#include <cstring>
#include <map>

namespace tflite {

ResourceVariable::ResourceVariable() {
  memset(&tensor_, 0, sizeof(TfLiteTensor));
}

ResourceVariable::ResourceVariable(ResourceVariable&& other) {
  tensor_ = other.tensor_;
  is_initialized_ = other.is_initialized_;

  memset(&other.tensor_, 0, sizeof(TfLiteTensor));
  other.is_initialized_ = false;
}

ResourceVariable::~ResourceVariable() {
  if (is_initialized_) {
    free(tensor_.data.raw);
    if (tensor_.dims) {
      TfLiteIntArrayFree(tensor_.dims);
    }
  }
}

TfLiteStatus ResourceVariable::AssignFrom(const TfLiteTensor* tensor) {
  // Save the old allocated resources and attributes that we might use.
  char* old_raw = tensor_.data.raw;
  size_t old_bytes = tensor_.bytes;
  TfLiteIntArray* old_dims = tensor_.dims;

  // Copy primitive parameters.
  memset(&tensor_, 0, sizeof(tensor_));
  tensor_.allocation_type = kTfLiteDynamic;
  tensor_.type = tensor->type;
  tensor_.params = tensor->params;
  tensor_.quantization = tensor->quantization;

  // Copy old shape if possible otherwise create a new one.
  if (TfLiteIntArrayEqual(old_dims, tensor->dims)) {
    tensor_.dims = old_dims;
  } else {
    TfLiteIntArrayFree(old_dims);
    tensor_.dims = TfLiteIntArrayCopy(tensor->dims);
  }

  // Reuse the same buffer if possible otherwise allocate a new one.
  tensor_.data.raw = old_raw;
  if (old_bytes != tensor->bytes) {
    TfLiteTensorRealloc(tensor->bytes, &tensor_);
  }

  memcpy(tensor_.data.raw, tensor->data.raw, tensor_.bytes);
  is_initialized_ = true;

  return kTfLiteOk;
}

}  // namespace tflite
