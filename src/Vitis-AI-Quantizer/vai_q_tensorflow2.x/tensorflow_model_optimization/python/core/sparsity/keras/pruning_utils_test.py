# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for utility functions in pruning_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import g3
from absl.testing import parameterized

import tensorflow as tf
from tensorflow_model_optimization.python.core.keras import compat
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_utils

glorot_uniform_initializer = tf.keras.initializers.glorot_uniform


@parameterized.named_parameters(
    ("1x1", [1, 1]), ("4x4", [4, 4]), ("6x6", [6, 6]), ("1x4", [1, 4]),
    ("4x1", [4, 1]), ("1x8", [1, 8]), ("8x1", [8, 1]))
class PruningUtilsParameterizedTest(tf.test.TestCase, parameterized.TestCase):

  def _compare_pooling_methods(self, weights, pooling_kwargs):
    with self.cached_session():
      compat.initialize_variables(self)
      pooled_weights_tf = tf.squeeze(
          tf.nn.pool(
              tf.reshape(
                  weights,
                  [1, weights.get_shape()[0],
                   weights.get_shape()[1], 1]), **pooling_kwargs))
      pooled_weights_factorized_pool = pruning_utils.factorized_pool(
          weights, **pooling_kwargs)
      self.assertAllClose(self.evaluate(pooled_weights_tf),
                          self.evaluate(pooled_weights_factorized_pool))

  def _compare_expand_tensor_with_kronecker_product(self, tensor, block_dim):
    with self.cached_session() as session:
      compat.initialize_variables(self)
      expanded_tensor = pruning_utils.expand_tensor(tensor, block_dim)
      kronecker_product = pruning_utils.kronecker_product(
          tensor, tf.ones(block_dim))
      expanded_tensor_val, kronecker_product_val = session.run(
          [expanded_tensor, kronecker_product])
      self.assertAllEqual(expanded_tensor_val, kronecker_product_val)

  def testFactorizedAvgPool(self, window_shape):
    shape = [1024, 2048]
    weights = tf.Variable(
        glorot_uniform_initializer()(shape), shape=shape, name="weights")
    pooling_kwargs = {
        "window_shape": window_shape,
        "pooling_type": "AVG",
        "strides": window_shape,
        "padding": "SAME"
    }
    self._compare_pooling_methods(weights, pooling_kwargs)

  def testFactorizedMaxPool(self, window_shape):
    shape = [1024, 2048]
    weights = tf.Variable(
        glorot_uniform_initializer()(shape), shape=shape, name="weights")
    pooling_kwargs = {
        "window_shape": window_shape,
        "pooling_type": "MAX",
        "strides": window_shape,
        "padding": "SAME"
    }
    self._compare_pooling_methods(weights, pooling_kwargs)

  def testExpandTensor(self, block_dim):
    weights = tf.random.normal(shape=[1024, 512])
    self._compare_expand_tensor_with_kronecker_product(weights, block_dim)


if __name__ == "__main__":
  tf.test.main()
