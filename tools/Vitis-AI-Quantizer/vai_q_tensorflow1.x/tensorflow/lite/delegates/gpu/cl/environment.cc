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

#include "tensorflow/lite/delegates/gpu/cl/environment.h"

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/cl_kernel.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"

namespace tflite {
namespace gpu {
namespace cl {

namespace {
CalculationsPrecision GetPossiblePrecision(
    const CLDevice& gpu, CalculationsPrecision desired_precision) {
  if (!gpu.SupportsFP16() && desired_precision != CalculationsPrecision::F32) {
    return CalculationsPrecision::F32;
  }

  return desired_precision;
}

std::string GetKernelOneLayerTextureArray() {
  return R"(

__kernel void main_function(__write_only image2d_array_t dst) {
  int X = (int)(get_global_id(0));
  int Y = (int)(get_global_id(1));

  write_imagef(dst, (int4)(X, Y, 0, 0), (float4)(2.0, 2.0, 2.0, 2.0));
}
)";
}

// Some Adreno < 600 have bug with one layer texture array. b/131099086
// If we have one layer texture array and will write smt from kernel to this
// texture, we will get zeroes instead of actual values.
// The same kernel will work, if we use texture array with more than one layer.
// With help of this code we can detect this bug.
Status CheckKernelSupportOfOneLayerTextureArray(Environment* env,
                                                bool* result) {
  // No bug on Adreno 6xx
  if (env->device().GetInfo().adreno_info.gpu_version >= 600) {
    *result = true;
    return OkStatus();
  }
  CLKernel kernel;
  RETURN_IF_ERROR(CreateKernel(GetKernelOneLayerTextureArray(), "main_function",
                               env, &kernel));
  Tensor tensor;
  RETURN_IF_ERROR(CreateTensor(env->context(), env->device(), 4, 4, 4,
                               DataType::FLOAT32,
                               TensorStorageType::TEXTURE_ARRAY, &tensor));
  RETURN_IF_ERROR(kernel.SetMemory(0, tensor.GetMemoryPtr()));
  RETURN_IF_ERROR(env->queue()->DispatchImplicit(kernel, {4, 4, 1}, {4, 4, 1}));
  std::vector<float> cpu_data(64, 0.0f);
  RETURN_IF_ERROR(tensor.ReadDataBHWC(absl::MakeSpan(cpu_data), env->queue()));

  *result = true;
  for (int i = 0; i < 64; ++i) {
    if (cpu_data[i] != 2.0) {
      *result = false;
      break;
    }
  }
  return OkStatus();
}

Status CreateEnvironment(Environment* result, bool shared,
                         cl_context_properties egl_context,
                         cl_context_properties egl_display) {
  CLDevice gpu;
  RETURN_IF_ERROR(CreateDefaultGPUDevice(&gpu));

  CLContext context;
  if (shared) {
    RETURN_IF_ERROR(CreateCLGLContext(gpu, egl_context, egl_display, &context));
  } else {
    RETURN_IF_ERROR(CreateCLContext(gpu, &context));
  }
  CLCommandQueue queue;
  RETURN_IF_ERROR(CreateCLCommandQueue(gpu, context, &queue));
  ProfilingCommandQueue profiling_queue;
  RETURN_IF_ERROR(CreateProfilingCommandQueue(gpu, context, &profiling_queue));

  *result = Environment(std::move(gpu), std::move(context), std::move(queue),
                        std::move(profiling_queue));

  if (result->device().IsAdreno() && result->device().SupportsTextureArray()) {
    bool supports_one_layer;
    RETURN_IF_ERROR(
        CheckKernelSupportOfOneLayerTextureArray(result, &supports_one_layer));
    if (!supports_one_layer) {
      result->GetDevicePtr()->DisableOneLayerTextureArray();
    }
  }

  return OkStatus();
}
}  // namespace

Environment::Environment(CLDevice&& device, CLContext&& context,
                         CLCommandQueue&& queue,
                         ProfilingCommandQueue&& profiling_queue)
    : device_(std::move(device)),
      context_(std::move(context)),
      queue_(std::move(queue)),
      profiling_queue_(std::move(profiling_queue)) {}

Environment::Environment(Environment&& environment)
    : device_(std::move(environment.device_)),
      context_(std::move(environment.context_)),
      queue_(std::move(environment.queue_)),
      profiling_queue_(std::move(environment.profiling_queue_)),
      program_cache_(std::move(environment.program_cache_)) {}

Environment& Environment::operator=(Environment&& environment) {
  if (this != &environment) {
    device_ = std::move(environment.device_);
    context_ = std::move(environment.context_);
    queue_ = std::move(environment.queue_);
    profiling_queue_ = std::move(environment.profiling_queue_);
    program_cache_ = std::move(environment.program_cache_);
  }
  return *this;
}

void Environment::SetHighPerformance() const {
  // TODO(sorokin) use cl_perf_hint if available
}

void Environment::SetDefaultPerformance() const {
  // TODO(sorokin) use cl_perf_hint if available
}

void Environment::SetLowPerformance() const {
  // TODO(sorokin) use cl_perf_hint if available
}

std::vector<CalculationsPrecision> Environment::GetSupportedPrecisions() const {
  std::vector<CalculationsPrecision> precisions;
  for (CalculationsPrecision precision :
       {CalculationsPrecision::F32, CalculationsPrecision::F32_F16,
        CalculationsPrecision::F16}) {
    if (IsSupported(precision)) {
      precisions.push_back(precision);
    }
  }
  return precisions;
}

bool Environment::IsSupported(CalculationsPrecision precision) const {
  switch (precision) {
    case CalculationsPrecision::F32_F16:
    case CalculationsPrecision::F16:
      return device_.SupportsFP16();
    case CalculationsPrecision::F32:
      return true;
  }
}

std::vector<TensorStorageType> Environment::GetSupportedTextureStorages()
    const {
  std::vector<TensorStorageType> storage_types = {
      TensorStorageType::TEXTURE_2D};
  if (device_.SupportsTextureArray()) {
    storage_types.push_back(TensorStorageType::TEXTURE_ARRAY);
  }
  return storage_types;
}

std::vector<TensorStorageType> Environment::GetSupportedStorages() const {
  std::vector<TensorStorageType> storage_types = {TensorStorageType::TEXTURE_2D,
                                                  TensorStorageType::BUFFER};
  if (device_.SupportsTextureArray()) {
    storage_types.push_back(TensorStorageType::TEXTURE_ARRAY);
  }
  return storage_types;
}

TensorStorageType GetOptimalStorageType(const CLDevice& gpu) {
  TensorStorageType storage_type;
  if (gpu.vendor() != Vendor::QUALCOMM) {
    storage_type = TensorStorageType::BUFFER;
  } else {
    if (gpu.IsAdreno6xxOrHigher()) {
      storage_type = TensorStorageType::TEXTURE_ARRAY;
    } else {
      storage_type = TensorStorageType::TEXTURE_2D;
    }
  }

  return storage_type;
}

Status CreateDefaultEnvironment(Environment* result) {
  return CreateEnvironment(result, false, 0, 0);
}

Status CreateEnvironment(Environment* result) {
  return CreateEnvironment(result, false, 0, 0);
}

Status CreateGLCompatibleEnvironment(cl_context_properties egl_context,
                                     cl_context_properties egl_display,
                                     Environment* result) {
  return CreateEnvironment(result, true, egl_context, egl_display);
}

Status CreateKernel(const std::string& code, const std::string& function_name,
                    Environment* env, CLKernel* result) {
  return CreateKernel(code, function_name, {}, env, result);
}

Status CreateKernel(const std::string& code, const std::string& function_name,
                    const std::vector<CompilerOptions>& compiler_options,
                    Environment* env, CLKernel* result) {
  return env->program_cache()->GetOrCreateCLKernel(
      code, function_name, compiler_options, env->context(), env->device(),
      result);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
