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

#include "tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h"

#include <sstream>

#include "tensorflow/compiler/tf2tensorrt/utils/trt_allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/mutex.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

string CalibrationContext::TerminateCalibration() {
  mutex_lock l(mu_);
  if (terminated_) return calibration_table_;

  TRTInt8Calibrator* raw_calibrator = calibrator_.get();
  raw_calibrator->waitAndSetDone();
  terminated_ = true;

  // At this point the calibration thread `thr_` is woken up and can
  // transfer the ownership of `calibrator_` and `engine_` at any time, so
  // it's not safe to use `calibrator_` below, but we can still access it
  // using raw pointer.
  // TODO(laigd): make TRTEngineOp::AllocateCalibrationResources() a member
  // function of this class instead.

  thr_->join();
  calibration_table_ = raw_calibrator->getCalibrationTableAsString();
  return calibration_table_;
}

const absl::string_view kTfTrtContainerName = "TF-TRT";

Logger& TRTEngineCacheResource::GetLogger() {
  static Logger* logger = new Logger();
  return *logger;
}

TRTEngineCacheResource::TRTEngineCacheResource(OpKernelContext* ctx,
                                               size_t capacity)
    : cache_(capacity) {
  auto device = ctx->device();
  auto alloc = device->GetAllocator(AllocatorAttributes());
  if (!alloc) {
    LOG(ERROR) << "Can't find device allocator for gpu device "
               << device->name();
    allocator_ = nullptr;
  } else {
    allocator_.reset(new TRTDeviceAllocator(alloc));
  }
}

TRTEngineCacheResource::~TRTEngineCacheResource() {
  VLOG(1) << "Destroying TRTEngineCacheResource...";
}

string TRTEngineCacheResource::DebugString() const {
  std::stringstream oss;
  using std::dec;
  using std::endl;
  using std::hex;
  oss << "TRTEngineCacheResource: ";
  oss << "TRTBaseAllocator = " << hex << allocator_.get() << dec << ", ";
  oss << "LRUCache = " << hex << &cache_ << dec << endl;
  oss << "Containing " << cache_.size() << " entries: " << endl;
  for (const auto& item : cache_) {
    mutex_lock lock(item.second->mu);
    oss << TensorShapeUtils::ShapeListString(item.first) << ": " << hex
        << "ICudaEngine: " << item.second->cuda_engine.get() << ", "
        << "IExecutionContext: " << item.second->execution_context.get() << dec
        << endl;
  }
  return oss.str();
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
