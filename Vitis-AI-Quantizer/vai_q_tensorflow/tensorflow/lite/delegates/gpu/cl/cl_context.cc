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

#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_image_format.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::vector<cl_image_format> GetSupportedImage2DFormats(cl_context context,
                                                        cl_mem_flags flags) {
  cl_uint num_image_formats;
  cl_int error = clGetSupportedImageFormats(
      context, flags, CL_MEM_OBJECT_IMAGE2D, 0, nullptr, &num_image_formats);
  if (error != CL_SUCCESS) {
    return {};
  }

  std::vector<cl_image_format> result(num_image_formats);
  error = clGetSupportedImageFormats(context, flags, CL_MEM_OBJECT_IMAGE2D,
                                     num_image_formats, &result[0], nullptr);
  if (error != CL_SUCCESS) {
    return {};
  }
  return result;
}

Status CreateCLContext(const CLDevice& device,
                       cl_context_properties* properties, CLContext* result) {
  int error_code;
  cl_device_id device_id = device.id();
  cl_context context =
      clCreateContext(properties, 1, &device_id, nullptr, nullptr, &error_code);
  if (!context) {
    return UnknownError(absl::StrCat("Failed to create a compute context - ",
                                     CLErrorCodeToString(error_code)));
  }

  *result = CLContext(context);
  return OkStatus();
}

}  // namespace

CLContext::CLContext(cl_context context) : context_(context) {}

CLContext::CLContext(CLContext&& context) : context_(context.context_) {
  context.context_ = nullptr;
}

CLContext& CLContext::operator=(CLContext&& context) {
  if (this != &context) {
    Release();
    std::swap(context_, context.context_);
  }
  return *this;
}

CLContext::~CLContext() { Release(); }

void CLContext::Release() {
  if (context_) {
    clReleaseContext(context_);
    context_ = nullptr;
  }
}

bool CLContext::IsFloatTexture2DSupported(int num_channels, DataType data_type,
                                          cl_mem_flags flags) const {
  auto supported_formats = GetSupportedImage2DFormats(context_, flags);
  for (auto format : supported_formats) {
    if (format.image_channel_data_type == ToImageChannelType(data_type) &&
        format.image_channel_order == ToChannelOrder(num_channels)) {
      return true;
    }
  }

  return false;
}

Status CreateCLContext(const CLDevice& device, CLContext* result) {
  return CreateCLContext(device, nullptr, result);
}

Status CreateCLGLContext(const CLDevice& device,
                         cl_context_properties egl_context,
                         cl_context_properties egl_display, CLContext* result) {
  if (!device.SupportsExtension("cl_khr_gl_sharing")) {
    return UnavailableError("Device doesn't support CL-GL sharing.");
  }
  cl_context_properties platform =
      reinterpret_cast<cl_context_properties>(device.platform());
  cl_context_properties props[] = {CL_GL_CONTEXT_KHR,
                                   egl_context,
                                   CL_EGL_DISPLAY_KHR,
                                   egl_display,
                                   CL_CONTEXT_PLATFORM,
                                   platform,
                                   0};
  return CreateCLContext(device, props, result);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
