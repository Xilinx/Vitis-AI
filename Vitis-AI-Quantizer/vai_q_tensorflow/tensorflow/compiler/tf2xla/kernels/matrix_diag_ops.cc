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

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace {

// Reads or infers lower_diag_index and upper_diag_index from kernel's input
// parameter "k". Also validates their values.
std::pair<int64, int64> ProcessDiagIndex(XlaOpKernelContext* context) {
  int64 lower_diag_index = 0;
  int64 upper_diag_index = 0;
  TensorShape diag_index_shape = context->InputShape("k");

  // Wrapping OP_REQUIRES* macros with a function because they can "return;"
  // early (without values) which contradicts ProcessDiagIndex's signature.
  auto validate_diag_indices = [&]() {
    if (diag_index_shape.dims() == 0) {
      OP_REQUIRES_OK(context,
                     context->ConstantInputAsIntScalar("k", &lower_diag_index));
      upper_diag_index = lower_diag_index;
    } else {
      std::vector<int64> diag_index;
      OP_REQUIRES_OK(context,
                     context->ConstantInputAsIntVector("k", &diag_index));
      OP_REQUIRES(
          context, !diag_index.empty() && diag_index.size() <= 2,
          errors::InvalidArgument(
              "diag_index must have only one or two elements, received ",
              diag_index.size(), " elements."));
      lower_diag_index = diag_index[0];
      upper_diag_index =
          (diag_index.size() > 1) ? diag_index[1] : lower_diag_index;
    }
    OP_REQUIRES(
        context, lower_diag_index <= upper_diag_index,
        errors::InvalidArgument(
            "lower_diag_index must not be larger than upper_diag_index: ",
            lower_diag_index, " > ", upper_diag_index));
  };
  validate_diag_indices();
  return {lower_diag_index, upper_diag_index};
}

// Makes sure lower_diag_index and upper_diag_index are consistent with the
// input matrix size.
void ValidateDiagIndexWithOutputMatrixSize(XlaOpKernelContext* context,
                                           const int64 lower_diag_index,
                                           const int64 upper_diag_index,
                                           const int64 num_rows,
                                           const int64 num_cols) {
  // `lower_diag_index == 0` condition is added to handle matrix shape = 0.
  OP_REQUIRES(context,
              (-num_rows < lower_diag_index && lower_diag_index < num_cols) ||
                  lower_diag_index == 0,
              errors::InvalidArgument(
                  "lower_diag_index is out of bound: ", lower_diag_index,
                  " It must be between ", -num_rows, " and ", num_cols));
  OP_REQUIRES(context,
              (-num_rows < upper_diag_index && upper_diag_index < num_cols) ||
                  upper_diag_index == 0,
              errors::InvalidArgument(
                  "upper_diag_index is out of bound: ", upper_diag_index,
                  " It must be between ", -num_rows, " and ", num_cols));
  OP_REQUIRES(context, lower_diag_index <= upper_diag_index,
              errors::InvalidArgument(
                  "lower_diag_index must not be larger than upper_diag_index: ",
                  lower_diag_index, " > ", upper_diag_index));
}

// Kernel to set matrix diagonals.
xla::XlaOp SetMatrixDiag(const xla::XlaOp input, const xla::XlaOp diag,
                         const TensorShape& input_shape, const int64 diag_rank,
                         const int64 num_diags, const int64 lower_diag_index,
                         const int64 upper_diag_index, const int64 max_diag_len,
                         const int64 num_rows, const int64 num_cols) {
  // Creates a padding config.
  const int input_rank = input_shape.dims();
  xla::PaddingConfig padding_config;
  padding_config = xla::MakeNoPaddingConfig(input_rank - 1);

  // Processes one diagonal at a time:
  // 1) Extracts a single diagonal (diag_slice).
  // 2) Broadcasts its contents to fill the whole matrix (diag_broadcast).
  // 3) Masks diag_broadcast to get the right diagonal shape.
  //
  // XLA can fuse multiple Broadcasts and Selects so this shouldn't be slow.
  //
  // For example,
  //   diag = [[2, 3, 0], k = (-1, 1), and num_rows = 4.
  //           [4, 5, 6],
  //           [7, 8, 9]]
  // The expected output is [[4, 2, 0],
  //                         [7, 5, 4],
  //                         [0, 8, 6],
  //                         [0, 0, 9]]
  // The 1st diagonal is created by:
  // 1) Extracting diag_slice = [1, 2, 0].
  // 2) Padding the vector to be as long as num_rows,
  //      diag_slice = [1, 2, 0, 0],
  //    then broadcasting diag_slice row-wise to a full matrix,
  //      diag_broadcast = [[1, 1, 1],
  //                        [2, 2, 2],
  //                        [0, 0, 0],
  //                        [0, 0, 0]]
  //    The padding value can be anything because it will not appear in the
  //    results after masking. Here, we use zero.
  // 3) Masking diag_broadcast with a mask of the shape of the 1st diagonal.
  //      mask = [[0, 1, 0],  -->  output = [[x, 2, x],
  //              [0, 0, 1],                 [x, x, 3],
  //              [0, 0, 0],                 [x, x, x],
  //              [0, 0, 0]]                 [x, x, x]],
  //    where x denotes the existing input contents.
  std::vector<int64> broadcast_dimensions(input_rank - 1);
  absl::c_iota(broadcast_dimensions, 0);
  auto output = input;
  for (int64 diag_index = lower_diag_index; diag_index <= upper_diag_index;
       ++diag_index) {
    // Extracts a single diagonal.
    auto diag_slice = diag;
    if (num_diags > 1) {
      const int64 mapped_diag_index = upper_diag_index - diag_index;
      diag_slice = xla::Collapse(
          xla::SliceInDim(diag, mapped_diag_index, mapped_diag_index + 1, 1,
                          diag_rank - 2),
          {diag_rank - 2, diag_rank - 1});
    }

    // Pads if necessary. Always pad at the end because shorter diagonals in
    // the input come padded at the end.
    const int64 padding_length =
        ((diag_index <= 0) ? num_cols : num_rows) - max_diag_len;
    const xla::XlaOp zero = xla::ScalarLike(input, 0);
    if (padding_length > 0) {
      padding_config.mutable_dimensions(input_rank - 2)
          ->set_edge_padding_high(padding_length);
      diag_slice = xla::Pad(diag_slice, zero, padding_config);
    }

    // Broadcasts column-wise for subdiagonals; row-wise for superdiagonals.
    broadcast_dimensions.back() =
        (diag_index <= 0) ? input_rank - 1 : input_rank - 2;
    xla::XlaOp diag_broadcast = xla::BroadcastInDim(
        diag_slice, input_shape.dim_sizes(), broadcast_dimensions);
    const auto mask = xla::GetDiagonalMask(output, diag_index);
    output = xla::Select(mask, diag_broadcast, output);
  }
  return output;
}

}  // namespace

class MatrixDiagOp : public XlaOpKernel {
 public:
  explicit MatrixDiagOp(OpKernelConstruction* context) : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    OP_REQUIRES(
        context, context->num_inputs() >= 1,
        errors::InvalidArgument("MatrixDiag op must have at least one input"));
    const TensorShape diag_shape = context->InputShape(0);
    OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(diag_shape),
                errors::InvalidArgument("Expected >= 1 dims, got shape ",
                                        diag_shape.DebugString()));

    const DataType dtype = context->expected_output_dtype(0);
    const xla::XlaOp zero = XlaHelpers::Zero(context->builder(), dtype);

    // Initializes MatrixDiagV2-specific variables.
    // Input arguments providing the values of num_rows and num_cols can be
    // absent (-1) and will be inferred later.
    int64 lower_diag_index = 0;
    int64 upper_diag_index = 0;
    int64 num_rows = -1;
    int64 num_cols = -1;
    xla::XlaOp padding_value = zero;

    // MatrixDiag and MatrixDiagV2 both use this OpKernel. MatrixDiag only has
    // one input, so we have to check the number of inputs before reading
    // additional parameters for MatrixDiagV2.
    if (context->num_inputs() > 1) {
      std::tie(lower_diag_index, upper_diag_index) = ProcessDiagIndex(context);
      OP_REQUIRES_OK(context, context->ConstantInputAsIntScalar(2, &num_rows));
      OP_REQUIRES_OK(context, context->ConstantInputAsIntScalar(3, &num_cols));
      padding_value = context->Input(4);
    }

    // More size validations.
    const int64 diag_rank = diag_shape.dims();
    const int64 max_diag_len = diag_shape.dim_size(diag_rank - 1);
    const int64 num_diags = upper_diag_index - lower_diag_index + 1;
    OP_REQUIRES(
        context,
        num_diags == 1 || num_diags == diag_shape.dim_size(diag_rank - 2),
        errors::InvalidArgument(
            "The number of diagonals provided in the input does not "
            "match the lower_diag_index and upper_diag_index range."));
    const int64 min_num_rows = max_diag_len - std::min(upper_diag_index, 0LL);
    const int64 min_num_cols = max_diag_len + std::max(lower_diag_index, 0LL);
    OP_REQUIRES(context, num_rows == -1 || num_rows >= min_num_rows,
                errors::InvalidArgument("The number of rows is too small."));
    OP_REQUIRES(context, num_cols == -1 || num_cols >= min_num_cols,
                errors::InvalidArgument("The number of columns is too small."));

    // Infers num_rows and num_cols. If both are unknown, assume that the output
    // is square. Otherwise, use smallest possible values.
    if (num_rows == -1 && num_cols == -1) {
      num_rows = std::max(min_num_rows, min_num_cols);
      num_cols = num_rows;
    } else if (num_rows == -1) {
      num_rows = min_num_rows;
    } else if (num_cols == -1) {
      num_cols = min_num_cols;
    }

    // At least one of num_rows and num_cols must match its minimum length.
    // Otherwise, we'll have some incomplete diagonals.
    OP_REQUIRES(context, num_rows == min_num_rows || num_cols == min_num_cols,
                errors::InvalidArgument(
                    "The number of rows or columns is not consistent with "
                    "the specified d_lower, d_upper, and diagonal."));

    // Actual processing.
    // Initializes the output tensor with padding_value.
    TensorShape output_shape = diag_shape;
    output_shape.RemoveLastDims((num_diags == 1) ? 1 : 2);
    output_shape.AddDim(num_rows);
    output_shape.AddDim(num_cols);
    xla::XlaOp output = xla::Broadcast(padding_value, output_shape.dim_sizes());
    xla::XlaOp diag = context->Input(0);
    context->SetOutput(
        0, SetMatrixDiag(output, diag, output_shape, diag_rank, num_diags,
                         lower_diag_index, upper_diag_index, max_diag_len,
                         num_rows, num_cols));
  }
};

REGISTER_XLA_OP(Name("MatrixDiag"), MatrixDiagOp);
REGISTER_XLA_OP(Name("MatrixDiagV2")
                    .CompileTimeConstantInput("k")
                    .CompileTimeConstantInput("num_rows")
                    .CompileTimeConstantInput("num_cols")
                    .CompileTimeConstantInput("padding_value"),
                MatrixDiagOp);

class MatrixDiagPartOp : public XlaOpKernel {
 public:
  explicit MatrixDiagPartOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    const TensorShape input_shape = context->InputShape(0);
    const int input_rank = input_shape.dims();

    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input_shape),
                errors::InvalidArgument(
                    "input must be at least 2-dim, received shape: ",
                    input_shape.DebugString()));

    const DataType dtype = context->expected_output_dtype(0);
    const xla::XlaOp zero = XlaHelpers::Zero(context->builder(), dtype);

    // Initializes MatrixDiagPartV2-specific variables.
    int64 lower_diag_index = 0;
    int64 upper_diag_index = 0;
    xla::XlaOp padding_value = zero;

    // MatrixDiagPart and MatrixDiagPartV2 both use this OpKernel.
    // MatrixDiagPart only has one input, so we have to check the number of
    // inputs before reading additional parameters in MatrixDiagV2.
    if (context->num_inputs() > 1) {
      std::tie(lower_diag_index, upper_diag_index) = ProcessDiagIndex(context);
      padding_value = context->Input(2);
    }

    // Checks if diag sizes are consistent with input.
    const int64 num_rows = input_shape.dim_size(input_rank - 2);
    const int64 num_cols = input_shape.dim_size(input_rank - 1);
    ValidateDiagIndexWithOutputMatrixSize(context, lower_diag_index,
                                          upper_diag_index, num_rows, num_cols);

    // Creates output shape.
    TensorShape output_shape = input_shape;
    output_shape.RemoveLastDims(2);
    const int num_diags = upper_diag_index - lower_diag_index + 1;
    if (num_diags > 1) output_shape.AddDim(num_diags);
    const int32 max_diag_len =
        std::min(num_rows + std::min(upper_diag_index, 0LL),
                 num_cols - std::max(lower_diag_index, 0LL));
    output_shape.AddDim(max_diag_len);

    // Computes output.
    xla::XlaOp input = context->Input(0);
    std::vector<xla::XlaOp> diag_list;
    xla::PaddingConfig padding_config;
    if (num_diags == 1) {
      context->SetOutput(0, xla::GetMatrixDiagonal(input, upper_diag_index));
      return;
    }
    padding_config = xla::MakeNoPaddingConfig(input_rank - 1);
    for (int diag_index = upper_diag_index; diag_index >= lower_diag_index;
         --diag_index) {
      auto single_diag = xla::GetMatrixDiagonal(input, diag_index);
      const int64 diag_length =
          (diag_index >= 0) ? (num_cols - diag_index) : (num_rows + diag_index);
      const int64 padding_length = max_diag_len - diag_length;
      if (padding_length > 0) {
        padding_config.mutable_dimensions(input_rank - 2)
            ->set_edge_padding_high(padding_length);
        single_diag = xla::Pad(single_diag, padding_value, padding_config);
      }
      diag_list.emplace_back(single_diag);
    }
    auto concat =
        xla::ConcatInDim(context->builder(), diag_list, input_rank - 2);
    context->SetOutput(0, xla::Reshape(concat, output_shape.dim_sizes()));
  }
};

REGISTER_XLA_OP(Name("MatrixDiagPart"), MatrixDiagPartOp);
REGISTER_XLA_OP(Name("MatrixDiagPartV2")
                    .CompileTimeConstantInput("k")
                    .CompileTimeConstantInput("padding_value"),
                MatrixDiagPartOp);

class MatrixSetDiagOp : public XlaOpKernel {
 public:
  explicit MatrixSetDiagOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* context) override {
    const TensorShape input_shape = context->InputShape(0);
    const TensorShape diag_shape = context->InputShape(1);
    const int input_rank = input_shape.dims();
    const int diag_rank = diag_shape.dims();

    // Preliminary validation of sizes.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input_shape),
                errors::InvalidArgument(
                    "input must be at least 2-dim, received shape: ",
                    input_shape.DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(diag_shape),
                errors::InvalidArgument(
                    "diagonal must be at least 1-dim, received shape: ",
                    diag_shape.DebugString()));

    // MatrixSetDiag and MatrixSetDiagV2 both use this OpKernel. MatrixSetDiag
    // only has two inputs, so we have to check the number of inputs before
    // reading additional parameters in MatrixSetDiagV2.
    int64 lower_diag_index = 0;
    int64 upper_diag_index = 0;
    if (context->num_inputs() > 2) {
      std::tie(lower_diag_index, upper_diag_index) = ProcessDiagIndex(context);
    }

    // Checks if diag sizes are consistent with input.
    const int64 num_rows = input_shape.dim_size(input_rank - 2);
    const int64 num_cols = input_shape.dim_size(input_rank - 1);
    ValidateDiagIndexWithOutputMatrixSize(context, lower_diag_index,
                                          upper_diag_index, num_rows, num_cols);
    const Eigen::Index num_diags = upper_diag_index - lower_diag_index + 1;
    OP_REQUIRES(
        context,
        lower_diag_index == upper_diag_index ||
            (diag_shape.dim_size(input_rank - 2) == num_diags),
        errors::InvalidArgument("The number of diagonals provided in `diag` "
                                "is not consistent with `lower_diag_index` and "
                                "`upper_diag_index`"));

    TensorShape expected_diag_shape = input_shape;
    expected_diag_shape.RemoveLastDims(2);
    if (num_diags > 1) expected_diag_shape.AddDim(num_diags);
    const int32 max_diag_len =
        std::min(num_rows + std::min(upper_diag_index, 0LL),
                 num_cols - std::max(lower_diag_index, 0LL));
    expected_diag_shape.AddDim(max_diag_len);
    OP_REQUIRES(
        context, expected_diag_shape == diag_shape,
        errors::InvalidArgument(
            "Either first dimensions of diagonal don't match input.shape[:-2], "
            "or diagonal.shape[:-1] is not equal to the longests diagonal in "
            "range [lower_diag_index:upper_diag_index].\nInput shape: ",
            input_shape.DebugString(),
            "\nDiagonal shape: ", diag_shape.DebugString(),
            "\nExpected diagonal shape: ", expected_diag_shape.DebugString()));

    // Actual processing.
    xla::XlaOp input = context->Input(0);
    xla::XlaOp diag = context->Input(1);
    context->SetOutput(
        0, SetMatrixDiag(input, diag, input_shape, diag_rank, num_diags,
                         lower_diag_index, upper_diag_index, max_diag_len,
                         num_rows, num_cols));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(MatrixSetDiagOp);
};

REGISTER_XLA_OP(Name("MatrixSetDiag"), MatrixSetDiagOp);
REGISTER_XLA_OP(Name("MatrixSetDiagV2").CompileTimeConstantInput("k"),
                MatrixSetDiagOp);

}  // namespace tensorflow
