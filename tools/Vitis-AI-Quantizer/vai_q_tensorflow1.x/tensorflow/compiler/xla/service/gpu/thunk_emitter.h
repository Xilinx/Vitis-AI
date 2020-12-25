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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_THUNK_EMITTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_THUNK_EMITTER_H_

#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace gpu {

// Implements handling of GPU execution for HLO operations that are handed off
// to specialzied thunks that do not require code generation. Intended to be
// mixed into GPU emitters.
class ThunkEmitter {
 public:
  class EmissionContext {
   public:
    virtual void AddThunkToThunkSequence(std::unique_ptr<Thunk> thunk) = 0;
    virtual StatusOr<BufferAllocation::Slice> MaybeGetAllocationSlice(
        const HloInstruction& hlo, const ShapeIndex& index) const = 0;
    virtual int64 ByteSizeOf(const Shape& shape) const = 0;
    virtual const se::Platform* platform() const = 0;

    virtual ~EmissionContext() = default;
  };

  explicit ThunkEmitter(EmissionContext* context) : context_(context) {}

  Status HandleCustomCall(HloInstruction* custom_call);
  Status HandleFft(HloInstruction* fft);
  Status HandleTriangularSolve(HloInstruction* hlo);
  Status HandleInfeed(HloInstruction* xla_infeed);
  Status HandleOutfeed(HloInstruction* outfeed);

 private:
  EmissionContext* context_;

  void AddThunkToThunkSequence(std::unique_ptr<Thunk> thunk) {
    return context_->AddThunkToThunkSequence(std::move(thunk));
  }

  StatusOr<BufferAllocation::Slice> MaybeGetAllocationSlice(
      const HloInstruction& hlo, const ShapeIndex& index) const {
    return context_->MaybeGetAllocationSlice(hlo, index);
  }

  int64 ByteSizeOf(const Shape& shape) { return context_->ByteSizeOf(shape); }

  const se::Platform* platform() const { return context_->platform(); }

  BufferAllocation::Slice GetAllocationSlice(
      const HloInstruction& hlo, const ShapeIndex& index = {}) const {
    return MaybeGetAllocationSlice(hlo, index).ValueOrDie();
  }

  // Returns a FftThunk that calls cuFFT to implement `inst`.
  std::unique_ptr<Thunk> BuildFftThunk(const HloInstruction* inst);

  // Returns a CholeskyThunk that calls cuSolver to implement `inst`.
  std::unique_ptr<Thunk> BuildCholeskyThunk(const HloInstruction* inst);

  // Returns a TriangularSolveThunk that calls cuBlas to implement `inst`.
  std::unique_ptr<Thunk> BuildTriangularSolveThunk(const HloInstruction* inst);

  // Returns a GemmThunk that calls gemm to implement `inst`. The caller needs
  // to make sure `inst` outlives the lifetime of the returned Thunk object.
  std::unique_ptr<Thunk> BuildGemmThunk(const HloInstruction* inst);

  // Returns an InfeedThunk that performs a host-to-device memcpy to implement
  // `inst`.
  std::unique_ptr<Thunk> BuildInfeedThunk(const HloInstruction* inst);

  // Returns an OutfeedThunk that performs a device-to-host memcpy to implement
  // `inst`.
  std::unique_ptr<Thunk> BuildOutfeedThunk(const HloInstruction* inst);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_THUNK_EMITTER_H_
