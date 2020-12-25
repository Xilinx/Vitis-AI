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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_LIB_SVD_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_LIB_SVD_H_

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// The singular value decomposition of a given matrix A[..., M, N], the original
// matrix is recovered by u * diag(d) * v_t, where the first dims(A) - 2
// dimensions are batch dimensions.
struct SVDResult {
  // The columns of U are the left-singular vectors, e.g.,
  // U[..., :, :]_T * U[..., :, :] = I.
  XlaOp u;
  // Vector(s) with the singular values, within each vector sorted in descending
  // order. The first dims(D) - 1 dimensions have the same size as the batch
  // dimensions of A. And U[..., :, i] * D[..., i] = A[..., :, :] * V[..., :,
  // i].
  XlaOp d;
  // The columns of V are the right-singular vectors. e.g.,
  // V[..., :, :]_T * U[..., :, :] = I.
  XlaOp v;
};

// TODO(kuny): Add a bool flag that supports SVD with economy (reduced)
// representation, which is more memory efficient, especially in the case of
// tall-skinny matrices.
SVDResult SVD(XlaOp a, int64 max_iter = 100, float epsilon = 1e-6,
              PrecisionConfig::Precision precision = PrecisionConfig::HIGHEST);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_LIB_SVD_H_
