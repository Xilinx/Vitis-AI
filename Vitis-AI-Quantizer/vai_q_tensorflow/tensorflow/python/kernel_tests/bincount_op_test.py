# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for math_ops.bincount."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class BincountTest(test_util.TensorFlowTestCase):

  def test_empty(self):
    with self.session(use_gpu=True):
      self.assertAllEqual(self.evaluate(math_ops.bincount([], minlength=5)),
                          [0, 0, 0, 0, 0])
      self.assertAllEqual(self.evaluate(math_ops.bincount([], minlength=1)),
                          [0])
      self.assertAllEqual(self.evaluate(math_ops.bincount([], minlength=0)),
                          [])
      self.assertEqual(self.evaluate(math_ops.bincount([], minlength=0,
                                                       dtype=np.float32)).dtype,
                       np.float32)
      self.assertEqual(self.evaluate(math_ops.bincount([], minlength=3,
                                                       dtype=np.float64)).dtype,
                       np.float64)

  def test_values(self):
    with self.session(use_gpu=True):
      self.assertAllEqual(self.evaluate(math_ops.bincount([1, 1, 1, 2, 2, 3])),
                          [0, 3, 2, 1])
      arr = [1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5]
      self.assertAllEqual(self.evaluate(math_ops.bincount(arr)),
                          [0, 5, 4, 3, 2, 1])
      arr += [0, 0, 0, 0, 0, 0]
      self.assertAllEqual(self.evaluate(math_ops.bincount(arr)),
                          [6, 5, 4, 3, 2, 1])

      self.assertAllEqual(self.evaluate(math_ops.bincount([])), [])
      self.assertAllEqual(self.evaluate(math_ops.bincount([0, 0, 0])), [3])
      self.assertAllEqual(self.evaluate(math_ops.bincount([5])),
                          [0, 0, 0, 0, 0, 1])
      self.assertAllEqual(self.evaluate(math_ops.bincount(np.arange(10000))),
                          np.ones(10000))

  def test_maxlength(self):
    with self.session(use_gpu=True):
      self.assertAllEqual(self.evaluate(math_ops.bincount([5], maxlength=3)),
                          [0, 0, 0])
      self.assertAllEqual(self.evaluate(math_ops.bincount([1], maxlength=3)),
                          [0, 1])
      self.assertAllEqual(self.evaluate(math_ops.bincount([], maxlength=3)),
                          [])

  def test_random_with_weights(self):
    num_samples = 10000
    with self.session(use_gpu=True):
      np.random.seed(42)
      for dtype in [dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64]:
        arr = np.random.randint(0, 1000, num_samples)
        if dtype == dtypes.int32 or dtype == dtypes.int64:
          weights = np.random.randint(-100, 100, num_samples)
        else:
          weights = np.random.random(num_samples)
        self.assertAllClose(
            self.evaluate(math_ops.bincount(arr, weights)),
            np.bincount(arr, weights))

  def test_random_without_weights(self):
    num_samples = 10000
    with self.session(use_gpu=True):
      np.random.seed(42)
      for dtype in [np.int32, np.float32]:
        arr = np.random.randint(0, 1000, num_samples)
        weights = np.ones(num_samples).astype(dtype)
        self.assertAllClose(
            self.evaluate(math_ops.bincount(arr, None)),
            np.bincount(arr, weights))

  def test_zero_weights(self):
    with self.session(use_gpu=True):
      self.assertAllEqual(
          self.evaluate(math_ops.bincount(np.arange(1000), np.zeros(1000))),
          np.zeros(1000))

  def test_negative(self):
    # unsorted_segment_sum will only report InvalidArgumentError on CPU
    with self.cached_session():
      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(math_ops.bincount([1, 2, 3, -1, 6, 8]))

  @test_util.run_deprecated_v1
  def test_shape_function(self):
    # size must be scalar.
    with self.assertRaisesRegexp(
        ValueError, "Shape must be rank 0 but is rank 1 for 'Bincount'"):
      gen_math_ops.bincount([1, 2, 3, -1, 6, 8], [1], [])
    # size must be positive.
    with self.assertRaisesRegexp(ValueError, "must be non-negative"):
      gen_math_ops.bincount([1, 2, 3, -1, 6, 8], -5, [])
    # if size is a constant then the shape is known.
    v1 = gen_math_ops.bincount([1, 2, 3, -1, 6, 8], 5, [])
    self.assertAllEqual(v1.get_shape().as_list(), [5])
    # if size is a placeholder then the shape is unknown.
    s = array_ops.placeholder(dtype=dtypes.int32)
    v2 = gen_math_ops.bincount([1, 2, 3, -1, 6, 8], s, [])
    self.assertAllEqual(v2.get_shape().as_list(), [None])


if __name__ == "__main__":
  googletest.main()
