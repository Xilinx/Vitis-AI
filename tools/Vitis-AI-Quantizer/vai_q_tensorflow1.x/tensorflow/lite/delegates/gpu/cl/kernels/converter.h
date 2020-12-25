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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVERTER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVERTER_H_

#include <memory>

#include "tensorflow/lite/delegates/gpu/cl/environment.h"
#include "tensorflow/lite/delegates/gpu/spi.h"

namespace tflite {
namespace gpu {
namespace cl {

class TensorObjectConverterBuilder {
 public:
  virtual ~TensorObjectConverterBuilder() = default;

  virtual bool IsSupported(const TensorObjectDef& input,
                           const TensorObjectDef& output) = 0;

  virtual Status MakeConverter(
      const TensorObjectDef& input, const TensorObjectDef& output,
      std::unique_ptr<TensorObjectConverter>* converter) = 0;
};

// Supports conversions from BHWC to internal OpenCL tensor representation and
// back. Also supports F16/F32.
std::unique_ptr<TensorObjectConverterBuilder> NewConverterBuilder(
    Environment* environment);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONVERTER_H_
