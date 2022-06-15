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
"""Tests for keras pruning wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

keras = tf.keras
layers = keras.layers
Prune = pruning_wrapper.PruneLowMagnitude


class PruningWrapperTest(tf.test.TestCase):

  def setUp(self):
    self.model = keras.Sequential()
    self.params = {
        'pruning_schedule': pruning_schedule.ConstantSparsity(0.5, 0),
        'block_size': (1, 1),
        'block_pooling_type': 'AVG'
    }

  def testPruneWrapperAllowsOnlyValidPoolingType(self):
    layer = layers.Dense(10)
    with self.assertRaises(ValueError):
      pruning_wrapper.PruneLowMagnitude(layer, block_pooling_type='MIN')

    pruning_wrapper.PruneLowMagnitude(layer, block_pooling_type='AVG')
    pruning_wrapper.PruneLowMagnitude(layer, block_pooling_type='MAX')

  def _check_mask_count(self, expected_mask_count=0):
    mask_count = 0
    for l in self.model.layers:
      mask_count += len(l.pruning_vars)
    self.assertEqual(mask_count, expected_mask_count)

  # TODO(suyoggupta): Randomize the layer dimensions
  def testDense(self):
    self.model.add(Prune(layers.Dense(10), **self.params))
    self.model.build(input_shape=(10, 1))

    self._check_mask_count(expected_mask_count=1)

  def testEmbedding(self):
    self.model.add(
        Prune(
            layers.Embedding(10, 10),
            input_shape=(10,),
            **self.params))
    self.model.build(input_shape=(10, 1))

    self._check_mask_count(expected_mask_count=1)

  def testConv2D(self):
    self.model.add(Prune(layers.Conv2D(4, (3, 3)), **self.params))
    self.model.build(input_shape=(1, 16, 16, 4))

    self._check_mask_count(expected_mask_count=1)

  def testPruneModel(self):
    self.model.add(Prune(layers.Conv2D(32, 5)))
    self.model.add(
        Prune(layers.MaxPooling2D((2, 2), (2, 2))))
    self.model.add(Prune(layers.Conv2D(64, 5)))
    self.model.add(
        Prune(layers.MaxPooling2D((2, 2), (2, 2))))
    self.model.add(Prune(layers.Flatten()))
    self.model.add(Prune(layers.Dense(1024)))
    self.model.add(Prune(layers.Dropout(0.4)))
    self.model.add(Prune(layers.Dense(10)))
    self.model.build(input_shape=(1, 28, 28, 1))

    self._check_mask_count(expected_mask_count=4)

    # Test serialization
    model_config = self.model.get_config()
    self.assertEqual(
        model_config,
        self.model.__class__.from_config(
            model_config,
            custom_objects={
                'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude
            }).get_config())


if __name__ == '__main__':
  tf.test.main()
