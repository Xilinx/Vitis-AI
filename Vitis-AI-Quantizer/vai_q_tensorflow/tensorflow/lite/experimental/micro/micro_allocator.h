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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_MICRO_ALLOCATOR_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_MICRO_ALLOCATOR_H_

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/experimental/micro/simple_tensor_allocator.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

// Allocator responsible for allocating memory for all intermediate tensors
// necessary to invoke a model.
class MicroAllocator {
 public:
  // The lifetime of the model, tensor allocator and error reporter must be at
  // least as long as that of the allocator object, since the allocator needs
  // them to be accessible during its entire lifetime.
  MicroAllocator(TfLiteContext* context, const Model* model,
                 uint8_t* tensor_arena, size_t arena_size,
                 ErrorReporter* error_reporter);

  // Specify a particular tensor as pre-allocated.  This means that this tensor
  // will internally point to the supplied buffer, and no new memory will be
  // provided.  The buffer must live at least as long as the allocator, since
  // the buffer will be used every time an op is invoked which uses the
  // specified tensor.  Most commonly this is useful when a platform-provided
  // DMA buffer is used as an input, and it is desirable to avoid unnecessarily
  // allocating a new buffer and copying from the DMA buffer. The user must
  // ensure the buffer is valid throughout each interpreter run, and is not
  // prematurely overwritten.
  TfLiteStatus RegisterPreallocatedInput(uint8_t* buffer, size_t input_index);

  // Run through the model and allocate all necessary input, output and
  // intermediate tensors except for those already provided via calls to
  // registerPreallocatedInput.
  TfLiteStatus AllocateTensors();

 private:
  const Model* model_;
  SimpleTensorAllocator tensor_allocator_;
  ErrorReporter* error_reporter_;
  TfLiteContext* context_;

  const SubGraph* subgraph_;
  const flatbuffers::Vector<flatbuffers::Offset<Operator>>* operators_;
  const flatbuffers::Vector<flatbuffers::Offset<Tensor>>* tensors_;
};

}  // namespace tflite
#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_MICRO_ALLOCATOR_H_
