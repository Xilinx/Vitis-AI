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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_ERRORS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_ERRORS_H_

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

// @return if error_code is success, then return OK status. Otherwise translates
// error code into a message.
inline Status GetOpenCLError(cl_int error_code) {
  if (error_code == CL_SUCCESS) {
    return OkStatus();
  }
  return InternalError("OpenCL error: " + CLErrorCodeToString(error_code));
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_ERRORS_H_
