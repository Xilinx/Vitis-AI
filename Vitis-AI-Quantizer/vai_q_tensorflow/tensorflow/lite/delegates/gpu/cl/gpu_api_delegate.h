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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_GPU_API_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_GPU_API_DELEGATE_H_

#include <stdint.h>

#include <EGL/egl.h>
#include <GLES3/gl31.h>
#include "tensorflow/lite/c/c_api_internal.h"

#ifdef SWIG
#define TFL_CAPI_EXPORT
#else
#if defined(_WIN32)
#ifdef TFL_COMPILE_LIBRARY
#define TFL_CAPI_EXPORT __declspec(dllexport)
#else
#define TFL_CAPI_EXPORT __declspec(dllimport)
#endif  // TFL_COMPILE_LIBRARY
#else
#define TFL_CAPI_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32
#endif  // SWIG

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

enum TfLiteGpuInferencePriority {
  TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION = 0,
  TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY = 1,
};

// Shader compilation options.
struct TFL_CAPI_EXPORT TfLiteGpuCompileOptions_New {
  // When set to zero, computations are carried out in 32-bit floating point.
  // Otherwise, the GPU may quantify tensors, downcast values, process in FP16
  // (recommended).
  int32_t precision_loss_allowed;

  // Priority is defined in TfLiteGpuInferencePriority.
  int32_t inference_priority;
};

struct TFL_CAPI_EXPORT TfLiteGpuDelegateOptions_New {
  TfLiteGpuCompileOptions_New compile_options;

  // [Optional]
  // Whenever EGL display and EGL context are set, corresponding OpenCL context
  // will be created.
  // These variables are required when using GL objects as inputs or outputs.
  EGLDisplay egl_display;
  EGLContext egl_context;

  // [Optional]
  // Contains data returned from TfLiteGpuDelegateGetSerializedBinaryCache call.
  // Invalid or incompatible data will be discarded. Compiled binary may become
  // incompatible when GPU driver is updated.
  const uint8_t* serialized_binary_cache_data;
  size_t serialized_binary_cache_size;
};

// Creates a new delegate instance that need to be destroyed with
// TfLiteGpuDelegateDelete_New when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the following default values are used:
// .compile_options = {
//   .precision_loss_allowed = false,
// }
// .egl_display = eglGetCurrentDisplay(),
// .egl_context = eglGetCurrentContext();
TFL_CAPI_EXPORT TfLiteDelegate* TfLiteGpuDelegateCreate_New(
    const TfLiteGpuDelegateOptions_New* options);

// Destroys a delegate created with `TfLiteGpuDelegateCreate_New` call.
TFL_CAPI_EXPORT void TfLiteGpuDelegateDelete_New(TfLiteDelegate* delegate);

enum TfLiteGpuDataLayout {
  TFLITE_GPU_DATA_LAYOUT_BHWC = 0,
  TFLITE_GPU_DATA_LAYOUT_DHWC4 = 1,
};

// Binds GL shader storage object to an input or an output tensor in the
// initialized delegate. Bound buffer should have sufficient storage to
// accommodate all elements of a tensor.
//
// Supports data of kTfliteFloat16 or kTfliteFloat32 types in BHWC or DHWC4 data
// layouts.
//
// *** Must be called *before* `Interpreter::ModifyGraphWithDelegate`. ***
TFL_CAPI_EXPORT TfLiteStatus TfLiteGpuDelegateBindGlBufferToTensor(
    TfLiteDelegate* delegate, GLuint buffer_id, int tensor_index,
    TfLiteType data_type, TfLiteGpuDataLayout data_layout);

// Returns opaque binary blob that contains a collection of cached OpenCL
// binaries. Returned data could be re-used later to speed up initialization
// time when new delegate is created for the same model.
// Returned data is valid only if used on the same device, otherwise it will
// not be compatible and will be discarded.
TFL_CAPI_EXPORT bool TfLiteGpuDelegateGetSerializedBinaryCache(
    TfLiteDelegate* delegate, size_t* size, const uint8_t** data);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_GPU_API_DELEGATE_H_
