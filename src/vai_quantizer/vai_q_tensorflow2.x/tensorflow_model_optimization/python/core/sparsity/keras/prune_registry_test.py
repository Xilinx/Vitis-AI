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
"""Tests for prune registry."""

import tensorflow as tf

from tensorflow_model_optimization.python.core.sparsity.keras import prunable_layer
from tensorflow_model_optimization.python.core.sparsity.keras import prune_registry

keras = tf.keras
layers = keras.layers
PruneRegistry = prune_registry.PruneRegistry


class PruneRegistryTest(tf.test.TestCase):

  class CustomLayer(layers.Layer):
    pass

  class CustomLayerFromPrunableLayer(layers.Dense):
    pass

  class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
      self.units = units
      self.state_size = units
      super(PruneRegistryTest.MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
      self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                    initializer='uniform',
                                    name='kernel')
      self.recurrent_kernel = self.add_weight(
          shape=(self.units, self.units),
          initializer='uniform',
          name='recurrent_kernel')
      self.built = True

    def call(self, inputs, states):
      prev_output = states[0]
      h = keras.backend.dot(inputs, self.kernel)
      output = h + keras.backend.dot(prev_output, self.recurrent_kernel)
      return output, [output]

  class MinimalRNNCellPrunable(MinimalRNNCell, prunable_layer.PrunableLayer):

    def get_prunable_weights(self):
      return [self.kernel, self.recurrent_kernel]

  def testSupportsKerasPrunableLayer(self):
    self.assertTrue(PruneRegistry.supports(layers.Dense(10)))

  def testSupportsKerasPrunableLayerAlias(self):
    # layers.Conv2D maps to layers.convolutional.Conv2D
    self.assertTrue(PruneRegistry.supports(layers.Conv2D(10, 5)))

  def testSupportsKerasNonPrunableLayer(self):
    # Dropout is a layer known to not be prunable.
    self.assertTrue(PruneRegistry.supports(layers.Dropout(0.5)))

  def testDoesNotSupportKerasUnsupportedLayer(self):
    # ConvLSTM2D is a built-in keras layer but not supported.
    self.assertFalse(PruneRegistry.supports(layers.ConvLSTM2D(2, (5, 5))))

  def testSupportsKerasRNNLayers(self):
    self.assertTrue(PruneRegistry.supports(layers.LSTM(10)))
    self.assertTrue(PruneRegistry.supports(layers.GRU(10)))
    self.assertTrue(PruneRegistry.supports(layers.SimpleRNN(10)))

  def testSupportsKerasRNNLayerWithRNNCellsParams(self):
    self.assertTrue(PruneRegistry.supports(layers.RNN(layers.LSTMCell(10))))

    self.assertTrue(
        PruneRegistry.supports(
            layers.RNN([
                layers.LSTMCell(10),
                layers.GRUCell(10),
                keras.experimental.PeepholeLSTMCell(10),
                layers.SimpleRNNCell(10)
            ])))

  def testDoesNotSupportKerasRNNLayerUnknownCell(self):
    self.assertFalse(PruneRegistry.supports(
        keras.layers.RNN(PruneRegistryTest.MinimalRNNCell(32))))

  def testSupportsKerasRNNLayerPrunableCell(self):
    self.assertTrue(PruneRegistry.supports(
        keras.layers.RNN(PruneRegistryTest.MinimalRNNCellPrunable(32))))

  def testDoesNotSupportCustomLayer(self):
    self.assertFalse(PruneRegistry.supports(PruneRegistryTest.CustomLayer(10)))

  def testDoesNotSupportCustomLayerInheritedFromPrunableLayer(self):
    self.assertFalse(
        PruneRegistry.supports(
            PruneRegistryTest.CustomLayerFromPrunableLayer(10)))

  def testMakePrunableRaisesErrorForKerasUnsupportedLayer(self):
    with self.assertRaises(ValueError):
      PruneRegistry.make_prunable(layers.ConvLSTM2D(2, (5, 5)))

  def testMakePrunableRaisesErrorForCustomLayer(self):
    with self.assertRaises(ValueError):
      PruneRegistry.make_prunable(PruneRegistryTest.CustomLayer(10))

  def testMakePrunableRaisesErrorForCustomLayerInheritedFromPrunableLayer(self):
    with self.assertRaises(ValueError):
      PruneRegistry.make_prunable(
          PruneRegistryTest.CustomLayerFromPrunableLayer(10))

  def testMakePrunableWorksOnKerasPrunableLayer(self):
    layer = layers.Dense(10)
    with self.assertRaises(AttributeError):
      layer.get_prunable_weights()

    PruneRegistry.make_prunable(layer)
    # Required since build method sets up the layer weights.
    keras.Sequential([layer]).build(input_shape=(10, 1))

    self.assertEqual([layer.kernel], layer.get_prunable_weights())

  def testMakePrunableWorksOnKerasNonPrunableLayer(self):
    layer = layers.Dropout(0.5)
    with self.assertRaises(AttributeError):
      layer.get_prunable_weights()

    PruneRegistry.make_prunable(layer)

    self.assertEqual([], layer.get_prunable_weights())

  def testMakePrunableWorksOnKerasRNNLayer(self):
    layer = layers.LSTM(10)
    with self.assertRaises(AttributeError):
      layer.get_prunable_weights()

    PruneRegistry.make_prunable(layer)
    keras.Sequential([layer]).build(input_shape=(2, 3, 4))

    self.assertEqual(
        [layer.cell.kernel, layer.cell.recurrent_kernel],
        layer.get_prunable_weights())

  def testMakePrunableWorksOnKerasRNNLayerWithRNNCellsParams(self):
    cell1 = layers.LSTMCell(10)
    cell2 = layers.GRUCell(5)
    layer = layers.RNN([cell1, cell2])
    with self.assertRaises(AttributeError):
      layer.get_prunable_weights()

    PruneRegistry.make_prunable(layer)
    keras.Sequential([layer]).build(input_shape=(2, 3, 4))

    expected_weights = [
        cell1.kernel, cell1.recurrent_kernel, cell2.kernel,
        cell2.recurrent_kernel
    ]
    self.assertEqual(expected_weights, layer.get_prunable_weights())

  def testMakePrunableWorksOnKerasRNNLayerWithPrunableCell(self):
    cell1 = layers.LSTMCell(10)
    cell2 = PruneRegistryTest.MinimalRNNCellPrunable(5)
    layer = layers.RNN([cell1, cell2])
    with self.assertRaises(AttributeError):
      layer.get_prunable_weights()

    PruneRegistry.make_prunable(layer)
    keras.Sequential([layer]).build(input_shape=(2, 3, 4))

    expected_weights = [
        cell1.kernel, cell1.recurrent_kernel, cell2.kernel,
        cell2.recurrent_kernel
    ]
    self.assertEqual(expected_weights, layer.get_prunable_weights())

  def testMakePrunableRaisesErrorOnRNNLayersUnsupportedCell(self):
    with self.assertRaises(ValueError):
      PruneRegistry.make_prunable(layers.RNN(
          [layers.LSTMCell(10), PruneRegistryTest.MinimalRNNCell(5)]))


if __name__ == '__main__':
  tf.test.main()
