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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_TUNING_PARAMETERS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_TUNING_PARAMETERS_H_

#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"

namespace tflite {
namespace gpu {
namespace cl {

enum class TuningType { EXHAUSTIVE, FAST };

struct TuningParameters {
  ProfilingCommandQueue* queue;
  const DeviceInfo* info;
  TuningType tuning_type = TuningType::EXHAUSTIVE;
};

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_TUNING_PARAMETERS_H_
