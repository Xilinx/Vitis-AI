/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_VITISAI_VITISAI_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_VITISAI_VITISAI_DELEGATE_H_

#include "tensorflow/lite/c/c_api_internal.h"

namespace {

typedef void (*ErrorHandler)(const char*);

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct {
  // A null-terminated string specifying the execution target.
  char target[32];
} TfLiteVitisAIDelegateOptions;

// Returns a structure with the default VitisAI delegate options.
TfLiteVitisAIDelegateOptions TfLiteVitisAIDelegateOptionsDefault();

// Creates a new delegate instance that need to be destroyed with
// `TfLiteVitisAIDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the following default values are used:
TfLiteDelegate* TfLiteVitisAIDelegateCreate(
    const TfLiteVitisAIDelegateOptions* options);

// Destroys a delegate created with `TfLiteVitisAIDelegateCreate` call.
void TfLiteVitisAIDelegateDelete(TfLiteDelegate* delegate);

// Creates a new delegate instance that need to be destroyed with
// `tflite_plugin_destroy_delegate` when delegate is no longer used by TFLite.
// When `num_options` is set to `0`, the following default values are used:
TfLiteDelegate* tflite_plugin_create_delegate(
    char** options_keys,
    char** options_values,
    size_t num_options,
    ErrorHandler error_handler);

// Destroys a delegate created with `tflite_plugin_create_delegate` call.
void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_DELEGATES_VITISAI_VITISAI_DELEGATE_H_
