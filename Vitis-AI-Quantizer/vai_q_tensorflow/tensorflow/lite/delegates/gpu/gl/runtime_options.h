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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_RUNTIME_OPTIONS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_RUNTIME_OPTIONS_H_

namespace tflite {
namespace gpu {
namespace gl {

struct RuntimeOptions {
  RuntimeOptions()
      : reuse_internal_objects(true), bundle_readonly_objects(true) {}

  // If enabled triggers greedy algorithm to re-use internal buffers when
  // possible.
  // Keep this false when, for example, one need to analyze intermediate
  // results for debugging purposes.
  bool reuse_internal_objects;

  // If enabled all readonly objects will be bundled to create as few buffers or
  // textures as possible.
  bool bundle_readonly_objects;
};

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_RUNTIME_OPTIONS_H_
