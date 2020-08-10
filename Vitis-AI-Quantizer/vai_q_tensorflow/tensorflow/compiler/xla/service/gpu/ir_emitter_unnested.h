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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_UNNESTED_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_UNNESTED_H_

#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/sequential_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/gpu/thunk_emitter.h"
#include "tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h"
#include "tensorflow/compiler/xla/service/llvm_ir/kernel_tiling.h"

namespace xla {
namespace gpu {

// Emits LLVM IR for an "unnested computation".
//
// An unnested computation is an HloComputation which you run by executing one
// or more kernels for each HloInstruction it contains.  Examples of unnested
// computations:
//
//  - An HloModule's root computation,
//  - The body of an HLO while loop,
//  - The true/false computation of an HLO conditional.
//
// Note the opportunity for confusion -- the while loop's computation is nested
// within the root computation, but it's emitted using IrEmitterUnnested!  Don't
// think about it too hard.
//
// Examples of things that are not unnested computations:
//
//  - The reducer of a kReduce HLO.  This is emitted using IrEmitterNested.
//  - The body of a fusion node.  IrEmitterUnenested emits the relevant code
//    within a kernel function using FusedIrEmitter.  (FusedIrEmitter is not
//    really an IrEmitter, but is more an "IR generator generator".)
//
class IrEmitterUnnested : public IrEmitter,
                          private ThunkEmitter::EmissionContext {
 public:
  // Parameter block_contains_multi_tiles indicates whether a tile block
  // consists of multiple tiles or not. If the tile block contains only one
  // tile, there is no need to use atomic operation to accumulate a local result
  // to a global result to implement reduction.
  using TileGenerator =
      std::function<void(const llvm_ir::IrArray::Index& output_tile_origin,
                         absl::Span<llvm::Value* const> output_tile_bounds)>;
  // KernelCodegenInfo records the common information to support the code
  // generation for a kernel to process tensor elements by blocks. A block of
  // tensor elements may contain one or multiple tiles. The code generators that
  // generate code for tile elements or block prologue/epilogue refer to this
  // class in their prototypes. If the implementations of such code generators
  // require other information that are specific to the HLO instructions, the
  // implementations need to define and use derived classes of this class.
  class KernelCodegenInfo {
   public:
    explicit KernelCodegenInfo(llvm_ir::KernelMappingScheme* mapping_scheme)
        : mapping_scheme_(mapping_scheme),
          lane_id_(nullptr),
          index_ty_(nullptr) {}
    virtual ~KernelCodegenInfo() {}

    void SetLaneId(llvm::Value* v) { lane_id_ = v; }
    void SetIndexType(llvm::Type* t) { index_ty_ = t; }

    llvm::Value* GetLaneId() const { return lane_id_; }
    llvm_ir::KernelMappingScheme* GetKernelMappingScheme() const {
      return mapping_scheme_;
    }
    llvm::Type* GetIndexType() const { return index_ty_; }

   protected:
    llvm_ir::KernelMappingScheme* mapping_scheme_;
    llvm::Value* lane_id_;
    llvm::Type* index_ty_;
  };

  // A function object to prepare for the code generation for a tile block.
  using BlockPrologueGenerator =
      std::function<void(HloInstruction* hlo, KernelCodegenInfo* kernel_info)>;
  // A function object to finalize the code generation for a tile block.
  using BlockEpilogueGenerator =
      std::function<void(HloInstruction* hlo, KernelCodegenInfo* kernel_info)>;

  // A function object to generate code to process one element in a tile.
  //
  // hlo: the instruction for which the code is generated for.
  // index: the index for the first output element of the current thread.
  // y_loc: The y coordinate within a tile.
  // x_loc: The x coordinate within a tile.
  // kernel_info: Other information to support the kernel code generation.
  // x_iter_num: When a thread process N elements in the X dimension, x_iter_num
  //             has a value of 0..N-1 to identify the element being process.
  using EmitElementFunction = std::function<void(
      const llvm_ir::IrArray::Index& index, llvm::Value* y_loc,
      llvm::Value* x_loc, int64 x_iter_num)>;

  using TileElementGenerator = std::function<void(
      llvm::Value* y, llvm::Value* x, const llvm_ir::IrArray::Index& index,
      const string& loop_name, llvm::Value* tile_height,
      llvm::Value* tile_width, KernelSupportLibrary* ksl)>;

  // KernelCodeGenerator records the code generator objects that generate code
  // for tile elements or tile block prologue/epilogue.
  class KernelCodeGenerator {
   public:
    explicit KernelCodeGenerator(
        TileElementGenerator tile_element_generator,
        BlockPrologueGenerator block_prologue_generator =
            [](HloInstruction*, KernelCodegenInfo*) {},
        BlockEpilogueGenerator block_epilogue_generator =
            [](HloInstruction*, KernelCodegenInfo*) {})
        : tile_element_generator_(std::move(tile_element_generator)),
          block_prologue_generator_(std::move(block_prologue_generator)),
          block_epilogue_generator_(std::move(block_epilogue_generator)) {}

    const TileElementGenerator& GetTileElementGenerator() const {
      return tile_element_generator_;
    }
    const BlockPrologueGenerator& GetBlockPrologueGenerator() const {
      return block_prologue_generator_;
    }
    const BlockEpilogueGenerator& GetBlockEpilogueGenerator() const {
      return block_epilogue_generator_;
    }

   private:
    TileElementGenerator tile_element_generator_;
    BlockPrologueGenerator block_prologue_generator_;
    BlockEpilogueGenerator block_epilogue_generator_;
  };

  IrEmitterUnnested(const HloModuleConfig& hlo_module_config,
                    const HloComputation* hlo_computation,
                    IrEmitterContext* ir_emitter_context);
  IrEmitterUnnested(const IrEmitterUnnested&) = delete;
  IrEmitterUnnested& operator=(const IrEmitterUnnested&) = delete;

  // Transfers the ownship of thunk_sequence_ out.
  std::unique_ptr<ThunkSequence> ConsumeThunkSequence() {
    return std::move(thunk_sequence_);
  }

  Status DefaultAction(HloInstruction* hlo) override;

  // IrEmitterUnnested handles the following instructions differently from
  // IrEmitter. It also mixes in some special handling for custom kernels
  // via the ThunkEmitter.
  Status HandleCopy(HloInstruction* copy) override;
  Status HandleConditional(HloInstruction* conditional) override;
  Status HandleConvolution(HloInstruction* convolution) override;
  Status HandleCustomCall(HloInstruction* custom_call) override;
  Status HandleDot(HloInstruction* dot) override;
  Status HandleFft(HloInstruction* fft) override;
  Status HandleFusion(HloInstruction* fusion) override;
  Status HandleGetTupleElement(HloInstruction* get_tuple_element) override;
  Status HandleReduce(HloInstruction* reduce) override;
  Status HandleSelectAndScatter(HloInstruction* instruction) override;
  Status HandleTuple(HloInstruction* tuple) override;
  Status HandleWhile(HloInstruction* xla_while) override;
  Status HandleInfeed(HloInstruction* xla_infeed) override;
  Status HandleOutfeed(HloInstruction* outfeed) override;
  Status HandleRng(HloInstruction* random) override;
  Status HandleRngGetAndUpdateState(HloInstruction* rng_state) override;
  Status HandleScatter(HloInstruction* scatter) override;
  Status HandleSelect(HloInstruction* select) override;
  Status HandleSort(HloInstruction* sort) override;
  Status HandleTriangularSolve(HloInstruction* hlo) override;
  Status HandleTupleSelect(HloInstruction* tuple_select) override;
  Status HandleAllReduce(HloInstruction* crs) override;
  Status HandleAfterAll(HloInstruction* after_all) override;
  Status HandleReplicaId(HloInstruction* hlo) override;
  Status HandleCollectivePermute(HloInstruction* hlo) override;

  Status EmitTargetElementLoop(
      const HloInstruction& hlo,
      const llvm_ir::ElementGenerator& body_emitter) override;

  // Same as `EmitTargetElementLoop`, but in given `thunk` rather than
  // `LastThunk()`.
  Status EmitTargetElementLoopInThunk(
      const HloInstruction& hlo, const llvm_ir::ElementGenerator& body_emitter,
      KernelThunk* thunk);

  // Emits LLVM global variables corresponding to constant instructions.
  Status EmitConstantGlobals();

 private:
  // Add a owning Thunk object to the thunk sequence.
  void AddThunkToThunkSequence(std::unique_ptr<Thunk> thunk) override {
    thunk_sequence_->emplace_back(std::move(thunk));
  }

  // A convenient helper for calling BufferAssignment::GetUniqueSlice.
  StatusOr<BufferAllocation::Slice> MaybeGetAllocationSlice(
      const HloInstruction& hlo, const ShapeIndex& index) const override {
    return ir_emitter_context_->buffer_assignment().GetUniqueSlice(&hlo, index);
  }

  BufferAllocation::Slice GetAllocationSlice(
      const HloInstruction& hlo, const ShapeIndex& index = {}) const {
    return MaybeGetAllocationSlice(hlo, index).ConsumeValueOrDie();
  }

  int64 ByteSizeOf(const Shape& shape) const override {
    return llvm_ir::ByteSizeOf(
        shape, ir_emitter_context_->llvm_module()->getDataLayout());
  }

  const se::Platform* platform() const override {
    return ir_emitter_context_->platform();
  }

  // Builds the prototype of the IR kernel for `inst` and adds it to the module.
  // This kernel takes as arguments pointers to the given buffer allocations.
  llvm::Function* BuildKernelPrototype(
      const HloInstruction& inst,
      absl::Span<const BufferAllocation* const> args);

  // Helper for writing extra outputs from inside a reduce kernel.
  Status EmitExtraOutputsForReduce(
      const HloInstruction* unnested_hlo, const llvm_ir::IrArray::Index& index,
      bool use_linear_index,
      absl::Span<const std::pair<llvm_ir::ElementGenerator, ShapeIndex>>
          extra_output_gens);

  // Generates code for reduction to contiguous dimensions.
  //
  // Prerequisite: `IsReductionFromOrToContiguousDimensions(*unnested_hlo)`
  Status EmitReductionFromOrToContiguousDimensions(
      HloInstruction* unnested_hlo);

  // Computes the KernelMappingScheme for the reduce HLO and indicates whether
  // the reduction is a row reduction. For an un-fused reduce op, unnested_hlo
  // and first_reduce are the same instruction. For a kInput fusion,
  // unnested_hlo is the fusion instruction while first_reduce is the first
  // reduce op.
  std::tuple<llvm_ir::KernelMappingScheme, bool>
  ComputeMappingSchemeAndReductionKind(const HloInstruction* unnested_hlo,
                                       const HloInstruction* first_reduce);

  // Emits code for an in-place scatter, modifying `thunk`s launch dimensions in
  // the process. `scatter` may be fused, scatter indices are taken from
  // `scatter_indices_gen`, updates from`updates_gen`. The output buffer is
  // expected to have the operand values in it already.
  Status EmitScatter(Thunk* thunk, HloInstruction* scatter,
                     const llvm_ir::ElementGenerator& scatter_indices_gen,
                     const llvm_ir::ElementGenerator& updates_gen);

  // Returns true if a 0-2-1 tiling algorithm is already used to emit the kernel
  // for the hlo instruction.
  bool CheckAndEmitHloWithTile021(HloInstruction* hlo);
  // Emits a kernel for the hlo instruction using a 0-2-1 tiling algorithm and
  // returns the launch dimensions for the kernel. This is a helper to support
  // the implementation of CheckAndEmitHloWithTile021.
  LaunchDimensions EmitHlo021Tile(HloInstruction* hlo,
                                  absl::Span<const int64> reduced_output_dims,
                                  absl::Span<const int64> tiled_param_ids);
  // Emits a kernel for an unnested HLO instruction.
  LaunchDimensions EmitKernel(HloInstruction* unnested_hlo,
                              absl::Span<const int64> param_ids,
                              const KernelCodeGenerator& kernel_generator,
                              KernelCodegenInfo* kernel_info);

  void EmitBlock(KernelCodegenInfo* kernel_info, KernelSupportLibrary* ksl,
                 llvm::Type* index_ty, TileGenerator emit_one_tile);

  // Emits code to process a tensor element in a tile for the given kCopy HLO
  // that performs a 0-2-1 transpose.
  void EmitTileElementForCopy(
      HloInstruction* hlo, const llvm_ir::IrArray::Index& index,
      const KernelCodegenInfo* kernel_info, llvm::Value* y_loc,
      llvm::Value* x_loc, int64 x_iter_num,
      absl::Span<llvm::Value* const> param_shmem_buffers);

  // Emits code to process a tensor element in a tile for the given kLoop fusion
  // HLO containing parameters that are 0-2-1 transpose of its outputs.
  void EmitTileElementForFusion(
      HloInstruction* hlo, const llvm_ir::IrArray::Index& index,
      const KernelCodegenInfo* kernel_info, llvm::Value* y_loc,
      llvm::Value* x_loc, int64 x_iter_num,
      absl::Span<llvm::Value* const> param_shmem_buffers);

  // Emits code to process a tensor element in a tile for the given input hlo
  // that is either a unnested kReduce or a kInput fusion.
  void EmitTileElementForReduction(
      HloInstruction* unnested_hlo, const Shape& reduction_operand_shape,
      absl::Span<HloInstruction* const> output_instructions,
      const llvm_ir::IrArray::Index& index,
      const KernelCodegenInfo* kernel_info, int64 x_iter_num);
  // Prepares for the code generation for a tile block of a reduction kernel.
  void EmitPrologueForReduction(
      HloInstruction* unnested_hlo, KernelCodegenInfo* kernel_info,
      absl::Span<HloInstruction* const> output_instructions);
  void EmitPrologueForOneReduction(HloInstruction* unnested_hlo,
                                   HloInstruction* reduce_inst, int reduce_idx,
                                   KernelCodegenInfo* kernel_info,
                                   GpuElementalIrEmitter* elemental_emitter,
                                   ShapeIndex output_shape_index);
  // Wraps up the code generation for a tile block of a reduction kernel.
  void EmitEpilogueForReduction(
      HloInstruction* unnested_hlo, KernelCodegenInfo* kernel_info,
      absl::Span<const HloInstruction* const> reduce_instructions);
  // For each reducer, emits the shuffle-down loop to accumulate the partial
  // result to the global result.
  void EmitFullWarpShuffleDownLoopForAllReduces(
      absl::Span<HloComputation* const> reducers,
      absl::Span<llvm::AllocaInst* const> partial_result_addresses);

  // Returns a KernelThunk that invokes the kernel emitted for `inst`. The
  // caller needs to make sure `inst` outlives the lifetime of the returned
  // Thunk object. The kernel implementation will be unrolled if unroll_factor
  // is greater than one. 'implements_whole_instruction' specifies whether this
  // KernelThunk implements the whole 'inst' HloInstruction. In some cases
  // 'inst' will be implemented by a sequence of Thunks.
  std::unique_ptr<KernelThunk> BuildKernelThunk(
      const HloInstruction* inst, bool implements_whole_instruction,
      int unroll_factor = 1);

  // Returns a thunk that, given a reduce or select-and-scatter op, initializes
  // its memory to the appropriate initial value.
  StatusOr<std::unique_ptr<Thunk>> BuildInitializerThunk(
      HloInstruction* hlo, const ShapeIndex& index = {});

  // Returns a WhileThunk that invokes thunk sequences for 'condition' and
  // 'body' sub-computations of while instruction 'hlo'.
  std::unique_ptr<Thunk> BuildWhileThunk(const HloInstruction* hlo);

  // Returns a ForThunk which executes 'loop_limit' invocations of a thunk
  // sequence from the 'body' sub-computation of the while instruction 'hlo'.
  std::unique_ptr<Thunk> BuildForThunk(const HloInstruction* hlo,
                                       const int64 loop_limit);

  // Returns a ConditionalThunk which executes the thunk sequence for the
  // 'branch_computation' corresponding to the predicate/branch_index of the
  // given conditional instruction.
  std::unique_ptr<Thunk> BuildConditionalThunk(const HloInstruction* hlo);

  Status Postprocess(HloInstruction* hlo) override;

  // Returns the last generated thunk.
  Thunk* LastThunk() const { return thunk_sequence_->back().get(); }

  // The thunk sequence this IrEmitter generates for the input computation.
  std::unique_ptr<ThunkSequence> thunk_sequence_;

  // The HloComputation that this IrEmitter emits code for.
  const HloComputation* hlo_computation_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_UNNESTED_H_
