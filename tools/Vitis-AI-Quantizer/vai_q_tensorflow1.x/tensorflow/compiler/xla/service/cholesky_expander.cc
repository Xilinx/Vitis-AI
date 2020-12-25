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

#include "tensorflow/compiler/xla/service/cholesky_expander.h"

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/loops.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

namespace {

// The Cholesky–Banachiewicz algorithm. See
// https://en.wikipedia.org/wiki/Cholesky_decomposition#The_Cholesky–Banachiewicz_and_Cholesky–Crout_algorithms
// for a description.
//
// def cholesky_unblocked(a):
//   assert len(a.shape) == 2 and a.shape[-2] == a.shape[-1]
//   n = a.shape[-2]
//   l = np.zeros_like(a)
//   for j in xrange(n):
//     row = l[..., j, :j]
//     row_t = np.swapaxes(row, -1, -2)
//     l[..., j, j] = np.sqrt(a[..., j, j] - np.dot(row, row_t))
//     l[..., j+1:, j] = (a[..., j+1:, j] - np.dot(l[..., j+1:, :j], row_t)) /
//                       l[..., j, j]
//   return l
// Returns a (result, error) pair.
std::pair<XlaOp, XlaOp> CholeskyUnblocked(
    XlaOp a, PrecisionConfig::Precision precision) {
  XlaBuilder* builder = a.builder();
  auto result = [&]() -> StatusOr<std::pair<XlaOp, XlaOp>> {
    TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));
    const int n_dims = a_shape.rank();
    const int64 n = ShapeUtil::GetDimension(a_shape, -1);
    auto major_dims = AsInt64Slice(a_shape.dimensions())
                          .subspan(
                              /*pos=*/0,
                              /*len=*/n_dims - 2);

    XlaOp l = ZerosLike(a);

    // Construct the for loop body to iterate over rows.
    auto body_fn =
        [&](XlaOp i, absl::Span<const XlaOp> loop_vars,
            XlaBuilder* body_builder) -> StatusOr<std::vector<XlaOp>> {
      std::vector<int64> row_shape_dims(major_dims.begin(), major_dims.end());
      std::vector<int64> col_shape_dims(major_dims.begin(), major_dims.end());
      row_shape_dims.push_back(1);
      row_shape_dims.push_back(n);
      auto mask_zeros_row =
          Zeros(body_builder,
                ShapeUtil::MakeShape(a_shape.element_type(), row_shape_dims));

      col_shape_dims.push_back(n);
      col_shape_dims.push_back(1);
      auto mask_zeros_col =
          Zeros(body_builder,
                ShapeUtil::MakeShape(a_shape.element_type(), col_shape_dims));

      auto mask_range_row =
          Iota(body_builder, ShapeUtil::MakeShape(S32, row_shape_dims),
               /*iota_dimension=*/n_dims - 1);
      auto mask_range_col =
          Iota(body_builder, ShapeUtil::MakeShape(S32, col_shape_dims),
               /*iota_dimension=*/n_dims - 2);
      auto body_a = loop_vars[0];
      auto body_l = loop_vars[1];
      auto seen_error = loop_vars[2];

      // row = l[..., i, :i]
      // select the whole i-th row, then mask out all columns past i-1
      auto zero = ConstantR0<int32>(body_builder, 0);
      auto l_i = DynamicSliceInMinorDims(body_l, {i, zero}, {1, n});
      auto row = Select(Ge(mask_range_row, i), mask_zeros_row, l_i);
      // a[..., i, i]
      auto a_ii = DynamicSliceInMinorDims(body_a, {i, i}, {1, 1});
      // np.dot(row, np.swapaxes(row, -1, -2))
      auto diag_dot = BatchDot(row, false, row, true, precision);
      // l[..., i, i] = np.sqrt(a[..., i, i] - np.dot(row,
      //                                              np.swapaxes(row, -1, -2)))
      auto l_ii = a_ii - diag_dot;
      seen_error =
          Or(seen_error, Any(Or(Le(l_ii, ZerosLike(l_ii)), IsNan(l_ii))));
      l_ii = Sqrt(l_ii);

      // a[..., i+1:, i]
      // select the whole i-th column, then mask out all rows above i+1
      auto a_0i = DynamicSliceInMinorDims(body_a, {i}, {1});
      auto a_ip1i = Select(Le(mask_range_col, i), mask_zeros_col, a_0i);

      // l[..., i+1:, i] = (a[..., i+1:, i] - np.dot(l[..., i+1:, :i], r.T)) /
      //                   l[..., i, i]
      // The columns in [i, n] are zeroed out in `row`, so we just have to
      // zero out rows above i+1 after the BatchDot. np.dot(l[..., :, :i],
      // r.T)
      auto dot = BatchDot(body_l, false, row, true, precision);
      // np.dot(l[..., i+1:, :i], r.T)
      auto dot_ip1 = Select(Le(mask_range_col, i), mask_zeros_col, dot);

      body_l =
          DynamicUpdateSliceInMinorDims(body_l, (a_ip1i - dot_ip1) / l_ii, {i});
      // Assign the diagonal after the rest of the column because otherwise the
      // column assign will wrap around and overwrite the diagonal assign.
      body_l = DynamicUpdateSliceInMinorDims(body_l, l_ii, {i, i});

      return std::vector<XlaOp>{body_a, body_l, seen_error};
    };

    TF_ASSIGN_OR_RETURN(
        auto cholesky_while,
        ForEachIndex(n, S32, body_fn, {a, l, ConstantR0<bool>(builder, false)},
                     "unblocked", builder));

    return std::make_pair(cholesky_while[1], cholesky_while[2]);
  }();
  if (!result.ok()) {
    XlaOp error = builder->ReportError(result.status());
    return {error, error};
  }
  return result.ValueOrDie();
}

XlaOp BuildCholesky(XlaOp a, int64 block_size,
                    PrecisionConfig::Precision precision) {
  XlaBuilder* builder = a.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape a_shape, builder->GetShape(a));
    const int ndims = a_shape.rank();
    if (ndims < 2) {
      return InvalidArgument(
          "Argument to Cholesky must have rank >= 2; shape was %s",
          a_shape.ToString());
    }

    const int64 n = ShapeUtil::GetDimension(a_shape, -1);
    if (n != ShapeUtil::GetDimension(a_shape, -2)) {
      return InvalidArgument(
          "Argument to Cholesky must be batched square matrices; got shape %s",
          ShapeUtil::HumanString(a_shape));
    }

    if (primitive_util::IsComplexType(a_shape.element_type())) {
      return Unimplemented(
          "Complex types are not implemented in Cholesky; got shape %s",
          ShapeUtil::HumanString(a_shape));
    }

    if (block_size < 1) {
      return InvalidArgument(
          "block_size argument to Cholesky must be >= 1; got %d", block_size);
    }

    // Blocked left-looking Cholesky factorization.
    // Algorithm 1 from
    // Haidar, Azzam, et al. "High-performance Cholesky factorization for
    // GPU-only execution." Proceedings of General Purpose GPUs. ACM, 2017.
    XlaOp l = ZerosLike(a);
    XlaOp seen_error = ConstantR0<bool>(builder, false);
    for (int64 i = 0; i < n; i += block_size) {
      int64 k = std::min(block_size, n - i);
      if (i > 0) {
        // TODO(phawkins): consider implementing SYRK for the diagonal part of
        // the panel.
        // a[i:, i:i+k] -= np.dot(l[i:, :i], np.transpose(l[i:i+k, :i]))
        auto lhs = SliceInMinorDims(l, {i, 0}, {n, i});
        auto rhs = SliceInMinorDims(l, {i, 0}, {i + k, i});
        auto delta = BatchDot(lhs, false, rhs, true, precision);
        auto before = SliceInMinorDims(a, {i, i}, {n, i + k});
        a = UpdateSliceInMinorDims(a, before - delta, {i, i});
      }

      // l[i:i+k, i:i+k] = cholesky_unblocked(a[i:i+k, i:i+k])
      auto x = SliceInMinorDims(a, {i, i}, {i + k, i + k});
      XlaOp factorized;
      XlaOp factorized_error;
      std::tie(factorized, factorized_error) = CholeskyUnblocked(x, precision);
      seen_error = Or(seen_error, factorized_error);
      l = UpdateSliceInMinorDims(l, factorized, {i, i});

      if (i + k < n) {
        // l[i+k:, i:i+k] =
        //     trsm_right_transpose(l[i:i+k, i:i+k], a[i+k:, i:i+k])
        auto panel = SliceInMinorDims(a, {i + k, i}, {n, i + k});
        auto update =
            TriangularSolve(factorized, panel,
                            /*left_side=*/false,
                            /*lower=*/true,
                            /*unit_diagonal=*/false,
                            /*transpose_a=*/TriangularSolveOptions::TRANSPOSE);
        l = UpdateSliceInMinorDims(l, update, {i + k, i});
      }
    }
    return Select(seen_error,
                  FullLike(l, std::numeric_limits<float>::quiet_NaN()), l);
  });
}

}  // namespace

bool CholeskyExpander::InstructionMatchesPattern(HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kCholesky;
}

StatusOr<HloInstruction*> CholeskyExpander::ExpandInstruction(
    HloInstruction* instruction) {
  const CholeskyOptions& options = instruction->cholesky_options();
  const string name = absl::StrFormat(
      "xla.cholesky_%s_%s", instruction->operand(0)->shape().ToString(),
      options.lower() ? "lower" : "upper");

  HloModule* module = instruction->parent()->parent();

  HloComputation*& computation =
      computation_cache_.emplace(name, nullptr).first->second;
  if (!computation) {
    // Builds a new expansion.
    //
    // TODO(b/62327888): We do something unusual here: we build the computation
    // using the XlaBuilder API, which is nominally an XLA client API. We do
    // this because the external APIs for building complicated computations
    // (XlaBuilder) are much more ergonomic than the internal ones. As it turns
    // out, XlaBuilder isn't really a client API—what it does is build a
    // HloModuleProto protocol buffer, that we can then deserialize and clone
    // into our HloModule. Ideally we would avoid the protocol buffer step;
    // that is left as an exercise for future work.
    XlaBuilder builder(name);
    XlaOp a = Parameter(&builder, 0, instruction->operand(0)->shape(), "a");
    XlaOp l = BuildCholesky(MaybeTransposeInMinorDims(a, !options.lower()),
                            /*block_size=*/128,
                            /*precision=*/PrecisionConfig::HIGHEST);
    MaybeTransposeInMinorDims(l, !options.lower());

    TF_ASSIGN_OR_RETURN(XlaComputation xla_computation, builder.Build());

    TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                        xla_computation.GetProgramShape());
    HloModuleConfig config(program_shape);
    TF_ASSIGN_OR_RETURN(auto new_module, HloModule::CreateFromProto(
                                             xla_computation.proto(), config));
    HloCloneContext context(module);
    computation =
        module->DeepCloneComputation(new_module->entry_computation(), &context);
  }

  return instruction->parent()->AddInstruction(HloInstruction::CreateCall(
      instruction->shape(), instruction->operands(), computation));
}

}  // namespace xla
