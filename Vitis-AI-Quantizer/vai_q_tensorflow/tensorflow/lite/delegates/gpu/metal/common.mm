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

#include "tensorflow/lite/delegates/gpu/metal/common.h"

#import <Metal/Metal.h>

#include <Availability.h>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/status.h"

// Compile-time message: print define name and value.
#define VALUE_TO_STRING(x) #x
#define VALUE(x) VALUE_TO_STRING(x)
#define VAR_NAME_VALUE(var) #var "=" VALUE(var)

namespace tflite {
namespace gpu {
namespace metal {

id<MTLDevice> GetBestSupportedMetalDevice() { return MTLCreateSystemDefaultDevice(); }

int GetMacOsGpuVersion(id<MTLDevice> device) {
  const std::vector<std::pair<MTLFeatureSet, int>> features = {
#if defined(__MAC_10_11) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_10_11
    {MTLFeatureSet_macOS_GPUFamily1_v1, 1},
#endif
#if defined(__MAC_10_12) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_10_12
    {MTLFeatureSet_macOS_GPUFamily1_v2, 1},
#endif
#if defined(__MAC_10_13) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_10_13
    {MTLFeatureSet_macOS_GPUFamily1_v3, 1},
#endif
#if defined(__MAC_10_14) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_10_14
    {MTLFeatureSet_macOS_GPUFamily1_v4, 1},
    {MTLFeatureSet_macOS_GPUFamily2_v1, 2},
#endif
  };
  for (const auto& type : features) {
    if ([device supportsFeatureSet:type.first]) {
      return type.second;
    }
  }
  return 0;
}

Status CreateComputeProgram(id<MTLDevice> device, NSString* code, NSString* functionName,
                            NSDictionary<NSString*, NSString*>* macros,
                            id<MTLComputePipelineState>* program) {
  MTLCompileOptions* options = [[MTLCompileOptions alloc] init];

#if (defined(__MAC_10_14) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_10_14) ||      \
    (defined(__IPHONE_12_0) && __IPHONE_OS_VERSION_MIN_REQUIRED >= __IPHONE_12_0) || \
    (defined(__TVOS_12_0) && __TV_OS_VERSION_MIN_REQUIRED >= __TVOS_12_0)
  [options setLanguageVersion:MTLLanguageVersion2_1];
#elif (defined(__MAC_10_13) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_10_13) ||    \
    (defined(__IPHONE_11_0) && __IPHONE_OS_VERSION_MIN_REQUIRED >= __IPHONE_11_0) || \
    (defined(__TVOS_11_0) && __TV_OS_VERSION_MIN_REQUIRED >= __TVOS_11_0)
  [options setLanguageVersion:MTLLanguageVersion2_0];
#elif (defined(__MAC_10_12) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_10_12) ||    \
    (defined(__IPHONE_10_0) && __IPHONE_OS_VERSION_MIN_REQUIRED >= __IPHONE_10_0) || \
    (defined(__TVOS_10_0) && __TV_OS_VERSION_MIN_REQUIRED >= __TVOS_10_0)
  [options setLanguageVersion:MTLLanguageVersion1_2];
#elif (defined(__MAC_10_11) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_10_11) ||  \
    (defined(__IPHONE_9_0) && __IPHONE_OS_VERSION_MIN_REQUIRED >= __IPHONE_9_0) || \
    (defined(__TVOS_9_0) && __TV_OS_VERSION_MIN_REQUIRED >= __TVOS_9_0)
  [options setLanguageVersion:MTLLanguageVersion1_1];
#else
#pragma message(VAR_NAME_VALUE(__MAC_OS_X_VERSION_MIN_REQUIRED))
#pragma message(VAR_NAME_VALUE(__IPHONE_OS_VERSION_MIN_REQUIRED))
#pragma message(VAR_NAME_VALUE(__TV_OS_VERSION_MIN_REQUIRED))
#if !defined(TARGET_OS_SIMULATOR) || TARGET_OS_SIMULATOR == 0
// NOLINTBEGIN
#error \
    "The Metal delegate is not supported on current target SDK. Minimum supported os: iOS/tvOS 9.0, macOS 10.11"
// NOLINTEND
#endif
#endif

  [options setFastMathEnabled:YES];
  [options setPreprocessorMacros:macros];
  NSError* error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:code options:options error:&error];
  if (!library) {
    NSString* errorString =
        [NSString stringWithFormat:@"newLibraryWithSource: %@", [error localizedDescription]];
    return InternalError([errorString UTF8String]);
  }

  id<MTLFunction> function = [library newFunctionWithName:functionName];
  if (!function) {
    NSString* errorString =
        [NSString stringWithFormat:@"newFunctionWithName: %@", [error localizedDescription]];
    return InternalError([errorString UTF8String]);
  }

  *program = [device newComputePipelineStateWithFunction:function error:&error];
  if (!program) {
    NSString* errorString =
        [NSString stringWithFormat:@"newComputePipelineStateWithFunction error: %@",
                                   [error localizedDescription]];
    return InternalError([errorString UTF8String]);
  }
  return OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
