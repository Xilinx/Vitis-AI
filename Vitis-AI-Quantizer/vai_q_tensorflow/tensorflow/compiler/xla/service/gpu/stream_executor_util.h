/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_STREAM_EXECUTOR_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_STREAM_EXECUTOR_UTIL_H_

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/cuda/ptxas_utils.h"
#include "tensorflow/stream_executor/kernel_spec.h"

// Helper functions for interacting with StreamExecutor.

namespace xla {
namespace gpu {

// Returns true if the given StreamExecutor is for a Volta or newer nvidia GPU.
bool IsVoltaOrLater(const se::StreamExecutor& stream_exec);

// Returns (input, filter, output) XLA Layout protos given the StreamExecutor
// layouts.
StatusOr<std::tuple<Layout, Layout, Layout>>
StreamExecutorConvLayoutsToXlaLayouts(const ConvolutionDimensionNumbers& dnums,
                                      se::dnn::DataLayout input,
                                      se::dnn::FilterLayout filter,
                                      se::dnn::DataLayout output);

// Returns (input, filter, output) StreamExecutor layouts given the XLA layouts.
StatusOr<
    std::tuple<se::dnn::DataLayout, se::dnn::FilterLayout, se::dnn::DataLayout>>
XlaConvLayoutsToStreamExecutorLayouts(const ConvolutionDimensionNumbers& dnums,
                                      const Layout& input, const Layout& filter,
                                      const Layout& output);

// Generates and returns a unique lock per each provided executor.
// Guarantees that blocks of code both holding a lock for the same provided
// executor (as given by this function) will not be running concurrently.
//
// This is used to prevent other XLA instances from trying to autotune on a
// device while another thread is using it.
tensorflow::mutex_lock LockGpu(const se::StreamExecutor* stream_exec);

// Creates a kernel with a provided name, based from provided PTX in ptx.
// The kernel should be executed using the provided executor.
// The argument cubin_data represents compiled PTX and may be left empty.
//
// The canonical storage for both ptx and cubin_data should outlive
// the lifetime of the kernel.
StatusOr<std::unique_ptr<se::KernelBase>> CreateKernel(
    absl::string_view kernel_name, uint64 num_args, absl::string_view ptx,
    absl::Span<const uint8> cubin_data, se::StreamExecutor* stream_exec);

// Runs loaded kernel on the stream with the provided arguments.
Status ExecuteKernelOnStream(const se::KernelBase& kernel,
                             absl::Span<const se::DeviceMemoryBase> args,
                             int64 threads_per_block, int64 block_count,
                             se::Stream* stream);

// Create PtxCompilationOptions out of HloModuleConfig.
se::cuda::PtxCompilationOptions PtxOptsFromConfig(
    const HloModuleConfig& hlo_module_config);

// Initializes `buffer` with random data on `stream`.
// `rng_state` is an inout parameter for the pseudorandom generator state.
// `buffer_type` determines what buffer would be filled out with.
//
// Precondition: `buffer_type` is a floating point type, `rng_state` needs to be
// initalized to zero on the first use.
void InitializeFloatBuffer(se::Stream* stream, PrimitiveType buffer_type,
                           int64* rng_state, se::DeviceMemoryBase buffer);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_STREAM_EXECUTOR_UTIL_H_
