/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/pooling.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

// Create a diagonal / batch diagonal matrix with 'input' on the diagonal.
xla::XlaOp CreateDiagonal(xla::XlaOp input, int64 last_dim_size,
                          absl::Span<const int64> other_dims) {
  xla::XlaBuilder* builder = input.builder();
  // Create two matrices that have the following forms, and compare them:
  //
  // [[0, 0, 0, 0]            [[0, 1, 2, 3]
  //  [1, 1, 1, 1]             [0, 1, 2, 3]
  //  [2, 2, 2, 2]             [0, 1, 2, 3]
  //  [3, 3, 3, 3]]            [0, 1, 2, 3]]
  //
  // This produces a predicate matrix of the right size, with "true" on the
  // diagonal.
  xla::XlaOp iota = xla::Iota(builder, xla::S32, last_dim_size);
  xla::XlaOp iota_broadcast = xla::Broadcast(iota, {last_dim_size});
  xla::XlaOp mask = xla::Eq(iota_broadcast, iota, {0});

  // If this is a batched diagonal, broadcast the mask across the other
  // dimensions.
  if (!other_dims.empty()) {
    mask = xla::Broadcast(mask, other_dims);
  }

  // Broadcast the input, and then use the mask computed above to select the
  // diagonal:
  // e.g, in 2D:
  //         [[t, f, f]    [[1, 1, 1]    [[0, 0, 0]      [[1, 0, 0]
  // select(  [f, t, f]  ,  [4, 4, 4]  ,  [0, 0, 0]  ) =  [0, 4, 0]
  //          [f, f, t]]    [9, 9, 9]]    [0, 0, 0]]      [0, 0, 9]]
  //
  std::vector<int64> out_dim_sizes(other_dims.begin(), other_dims.end());
  out_dim_sizes.push_back(last_dim_size);
  out_dim_sizes.push_back(last_dim_size);

  // Broadcast into the second to last dimension.
  std::vector<int64> broadcast_dimensions(other_dims.size() + 1);
  absl::c_iota(broadcast_dimensions, 0);
  ++broadcast_dimensions.back();
  xla::XlaOp input_broadcast =
      xla::BroadcastInDim(input, out_dim_sizes, broadcast_dimensions);
  return xla::Select(mask, input_broadcast, xla::ZerosLike(input_broadcast));
}

class DiagOp : public XlaOpKernel {
 public:
  explicit DiagOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES(ctx, ctx->num_inputs() >= 1,
                errors::InvalidArgument("Diag op must have at an input"));
    const TensorShape input_shape = ctx->InputShape(0);

    auto dims = input_shape.dim_sizes();
    OP_REQUIRES(ctx, !dims.empty(),
                errors::InvalidArgument("Expected 1 <= dims, got shape ",
                                        input_shape.DebugString()));

    xla::XlaOp input = ctx->Input(0);

    // Picture:
    // tf.diag([1, 2, 3, 4]) ==> [[1, 0, 0, 0]
    //                            [0, 2, 0, 0]
    //                            [0, 0, 3, 0]
    //                            [0, 0, 0, 4]]

    // Flattens the input to 1D.
    int64 size = input_shape.num_elements();
    input = xla::Reshape(input, {size});

    // Create an R2 with the R1 diagonal.
    xla::XlaOp diag = CreateDiagonal(input, size, /*other_dims=*/{});

    // Reshapes to the final shape.
    std::vector<int64> new_dims(dims.size() * 2);
    std::copy(dims.begin(), dims.end(), new_dims.begin());
    std::copy(dims.begin(), dims.end(), new_dims.begin() + dims.size());
    diag = xla::Reshape(diag, new_dims);

    ctx->SetOutput(0, diag);
  }
};

REGISTER_XLA_OP(Name("Diag"), DiagOp);

class DiagPartOp : public XlaOpKernel {
 public:
  explicit DiagPartOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    auto dims = input_shape.dim_sizes();

    int num_dims = dims.size();
    const int out_dims = num_dims / 2;

    OP_REQUIRES(ctx, 2 <= num_dims,
                errors::InvalidArgument("Expected 2 <= dims, got shape ",
                                        input_shape.DebugString()));
    OP_REQUIRES(ctx, num_dims % 2 == 0,
                errors::InvalidArgument("The input tensor must have even rank; "
                                        "got shape ",
                                        input_shape.DebugString()));
    int64 new_size = 1;
    std::vector<int64> new_dims;
    for (int i = 0; i < out_dims; i++) {
      OP_REQUIRES(
          ctx, dims[i] == dims[i + out_dims],
          errors::InvalidArgument("Invalid shape ", input_shape.DebugString(),
                                  ": dimensions ", i, " and ", i + out_dims,
                                  " do not match."));
      new_size *= dims[i];
      new_dims.push_back(dims[i]);
    }

    xla::XlaOp input = ctx->Input(0);

    xla::XlaOp output = xla::Reshape(
        xla::GetMatrixDiagonal(xla::Reshape(input, {new_size, new_size})),
        new_dims);

    ctx->SetOutput(0, output);
  }
};

REGISTER_XLA_OP(Name("DiagPart"), DiagPartOp);

}  // namespace
}  // namespace tensorflow
