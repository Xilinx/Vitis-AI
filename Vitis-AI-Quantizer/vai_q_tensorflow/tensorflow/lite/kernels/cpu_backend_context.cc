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

#include "tensorflow/lite/kernels/cpu_backend_context.h"

#include "public/gemmlowp.h"
#include "tensorflow/lite/experimental/ruy/context.h"

namespace tflite {

CpuBackendContext* CpuBackendContext::GetFromContext(TfLiteContext* context) {
  auto* external_context = static_cast<ExternalCpuBackendContext*>(
      context->GetExternalContext(context, kTfLiteCpuBackendContext));

  if (external_context == nullptr) {
    TF_LITE_FATAL(
        "ExternalCpuBackendContext isn't properly initialized during TFLite "
        "interpreter initialization.");
  }

  auto* cpu_backend_context = static_cast<CpuBackendContext*>(
      external_context->internal_backend_context());
  if (cpu_backend_context == nullptr) {
    // We do the lazy initialization here for the TfLiteInternalBackendContext
    // that's wrapped inside ExternalCpuBackendContext.
    cpu_backend_context = new CpuBackendContext();
    if (context->recommended_num_threads != -1) {
      cpu_backend_context->SetMaxNumThreads(context->recommended_num_threads);
    }
    external_context->set_internal_backend_context(
        std::unique_ptr<TfLiteInternalBackendContext>(cpu_backend_context));
  }

  return cpu_backend_context;
}

CpuBackendContext::CpuBackendContext()
    : TfLiteInternalBackendContext(),
      ruy_context_(new ruy::Context),
      gemmlowp_context_(new gemmlowp::GemmContext) {
  SetMaxNumThreads(1);
}

CpuBackendContext::~CpuBackendContext() {}

void CpuBackendContext::SetMaxNumThreads(int max_num_threads) {
  max_num_threads_ = max_num_threads;
  ruy_context_->max_num_threads = max_num_threads;
  gemmlowp_context_->set_max_num_threads(max_num_threads);
}

}  // namespace tflite
