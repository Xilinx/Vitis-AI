/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_TOOLS_ACCURACY_ILSVRC_DEFAULT_CUSTOM_DELEGATES_H_
#define TENSORFLOW_LITE_TOOLS_ACCURACY_ILSVRC_DEFAULT_CUSTOM_DELEGATES_H_

#include <string>

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/tools/evaluation/stages/image_classification_stage.h"

namespace tflite {
namespace evaluation {

// Applies custom delegates on the provided ImageClassificationStage, if
// applicable.
TfLiteStatus ApplyCustomDelegates(const std::string& delegate,
                                  int num_interpreter_threads,
                                  ImageClassificationStage* stage_ptr) {
  return kTfLiteOk;
}

}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_ACCURACY_ILSVRC_DEFAULT_CUSTOM_DELEGATES_H_
