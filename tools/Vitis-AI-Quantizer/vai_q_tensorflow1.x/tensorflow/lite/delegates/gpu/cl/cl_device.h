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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_DEVICE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_DEVICE_H_

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

enum class Vendor { QUALCOMM, MALI, POWERVR, NVIDIA, UNKNOWN };
std::string VendorToString(Vendor v);

enum class OpenCLVersion { CL_1_0, CL_1_1, CL_1_2, CL_2_0 };
std::string OpenCLVersionToString(OpenCLVersion version);

// for use only in cl_device.cc, but putted here to make tests
int GetAdrenoGPUVersion(const std::string& gpu_version);

struct AdrenoInfo {
  AdrenoInfo() = default;
  explicit AdrenoInfo(const std::string& device_version);
  int gpu_version = -1;  // can be, for example, 405/430/540/530/630 etc.

  // This function returns some not very documented physical parameter of
  // Adreno6xx GPU.
  // We obtained it using Snapdragon Profiler.
  int GetMaximumWavesCount() const;

  // returns amount of register memory per CU(Compute Unit) in bytes.
  int GetRegisterMemorySizePerComputeUnit() const;

  // returns maximum possible amount of waves based on register usage.
  int GetMaximumWavesCount(int register_footprint_per_tread,
                           bool full_wave = true) const;

  int GetWaveSize(bool full_wave) const;

  // Not supported on some Adreno devices with specific driver version.
  // b/131099086
  bool support_one_layer_texture_array = true;
};

struct DeviceInfo {
  DeviceInfo() = default;
  explicit DeviceInfo(cl_device_id id);

  bool SupportsTextureArray() const;

  std::vector<std::string> extensions;
  bool supports_fp16;
  Vendor vendor;
  OpenCLVersion cl_version;
  int compute_units_count;
  int image2d_max_width;
  int image2d_max_height;
  int image_buffer_max_size;
  int image_array_max_layers;
  int3 max_work_group_sizes;

  AdrenoInfo adreno_info;
};

// A wrapper around opencl device id
class CLDevice {
 public:
  CLDevice() = default;
  CLDevice(cl_device_id id, cl_platform_id platform_id);

  CLDevice(CLDevice&& device);
  CLDevice& operator=(CLDevice&& device);
  CLDevice(const CLDevice&);
  CLDevice& operator=(const CLDevice&);

  ~CLDevice() {}

  cl_device_id id() const { return id_; }
  cl_platform_id platform() const { return platform_id_; }
  std::string GetPlatformVersion() const;

  const DeviceInfo& GetInfo() const { return info_; }
  const DeviceInfo* GetInfoPtr() const { return &info_; }

  Vendor vendor() const { return info_.vendor; }
  OpenCLVersion cl_version() const { return info_.cl_version; }
  bool SupportsFP16() const;
  bool SupportsTextureArray() const;
  bool SupportsExtension(const std::string& extension) const;
  bool IsAdreno() const;
  bool IsAdreno3xx() const;
  bool IsAdreno4xx() const;
  bool IsAdreno5xx() const;
  bool IsAdreno6xx() const;
  bool IsAdreno6xxOrHigher() const;

  // To track bug on some Adreno. b/131099086
  bool SupportsOneLayerTextureArray() const;
  void DisableOneLayerTextureArray();

 private:
  cl_device_id id_ = nullptr;
  cl_platform_id platform_id_ = nullptr;
  DeviceInfo info_;
};

Status CreateDefaultGPUDevice(CLDevice* result);

template <typename T>
T GetDeviceInfo(cl_device_id id, cl_device_info info) {
  T result;
  cl_int error = clGetDeviceInfo(id, info, sizeof(T), &result, nullptr);
  if (error != CL_SUCCESS) {
    return -1;
  }
  return result;
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_DEVICE_H_
