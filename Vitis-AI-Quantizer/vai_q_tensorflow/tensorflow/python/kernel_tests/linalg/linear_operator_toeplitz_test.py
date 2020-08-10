# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

import numpy as np
import scipy.linalg

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import spectral_ops_test_util
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.ops.linalg import linear_operator_toeplitz
from tensorflow.python.platform import test

linalg = linalg_lib

_to_complex = linear_operator_toeplitz._to_complex


class LinearOperatorToeplitzTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  @contextlib.contextmanager
  def _constrain_devices_and_set_default(self, sess, use_gpu, force_gpu):
    """We overwrite the FFT operation mapping for testing."""
    with test.TestCase._constrain_devices_and_set_default(
        self, sess, use_gpu, force_gpu) as sess:
      with spectral_ops_test_util.fft_kernel_label_map():
        yield sess

  def setUp(self):
    # TODO(srvasude): Lower these tolerances once specialized solve and
    # determinants are implemented.
    self._atol[dtypes.float32] = 1e-3
    self._rtol[dtypes.float32] = 1e-3
    self._atol[dtypes.float64] = 1e-10
    self._rtol[dtypes.float64] = 1e-10
    self._atol[dtypes.complex64] = 1e-3
    self._rtol[dtypes.complex64] = 1e-3
    self._atol[dtypes.complex128] = 1e-10
    self._rtol[dtypes.complex128] = 1e-10

  @staticmethod
  def skip_these_tests():
    # Skip solve tests, as these could have better stability
    # (currently exercises the base class).
    # TODO(srvasude): Enable these when solve is implemented.
    return ["cholesky", "inverse", "solve", "solve_with_broadcast"]

  @staticmethod
  def operator_shapes_infos():
    shape_info = linear_operator_test_util.OperatorShapesInfo
    # non-batch operators (n, n) and batch operators.
    return [
        shape_info((1, 1)),
        shape_info((1, 6, 6)),
        shape_info((3, 4, 4)),
        shape_info((2, 1, 3, 3))
    ]

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    shape = list(build_info.shape)
    row = np.random.uniform(low=1., high=5., size=shape[:-1])
    col = np.random.uniform(low=1., high=5., size=shape[:-1])

    # Make sure first entry is the same
    row[..., 0] = col[..., 0]

    if ensure_self_adjoint_and_pd:
      # Note that a Toeplitz matrix generated from a linearly decreasing
      # non-negative sequence is positive definite. See
      # https://www.math.cinvestav.mx/~grudsky/Papers/118_29062012_Albrecht.pdf
      # for details.
      row = np.linspace(start=10., stop=1., num=shape[-1])

      # The entries for the first row and column should be the same to guarantee
      # symmetric.
      row = col

    lin_op_row = math_ops.cast(row, dtype=dtype)
    lin_op_col = math_ops.cast(col, dtype=dtype)

    if use_placeholder:
      lin_op_row = array_ops.placeholder_with_default(
          lin_op_row, shape=None)
      lin_op_col = array_ops.placeholder_with_default(
          lin_op_col, shape=None)

    operator = linear_operator_toeplitz.LinearOperatorToeplitz(
        row=lin_op_row,
        col=lin_op_col,
        is_self_adjoint=True if ensure_self_adjoint_and_pd else None,
        is_positive_definite=True if ensure_self_adjoint_and_pd else None)

    flattened_row = np.reshape(row, (-1, shape[-1]))
    flattened_col = np.reshape(col, (-1, shape[-1]))
    flattened_toeplitz = np.zeros(
        [flattened_row.shape[0], shape[-1], shape[-1]])
    for i in range(flattened_row.shape[0]):
      flattened_toeplitz[i] = scipy.linalg.toeplitz(
          flattened_col[i],
          flattened_row[i])
    matrix = np.reshape(flattened_toeplitz, shape)
    matrix = math_ops.cast(matrix, dtype=dtype)

    return operator, matrix

  def test_scalar_row_col_raises(self):
    with self.assertRaisesRegexp(ValueError, "must have at least 1 dimension"):
      linear_operator_toeplitz.LinearOperatorToeplitz(1., 1.)

    with self.assertRaisesRegexp(ValueError, "must have at least 1 dimension"):
      linear_operator_toeplitz.LinearOperatorToeplitz([1.], 1.)

    with self.assertRaisesRegexp(ValueError, "must have at least 1 dimension"):
      linear_operator_toeplitz.LinearOperatorToeplitz(1., [1.])


if __name__ == "__main__":
  linear_operator_test_util.add_tests(LinearOperatorToeplitzTest)
  test.main()
