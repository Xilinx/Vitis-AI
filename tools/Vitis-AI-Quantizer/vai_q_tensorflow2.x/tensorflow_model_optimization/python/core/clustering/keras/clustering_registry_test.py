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
"""Tests for keras clustering registry API."""

import numpy as np
import tensorflow as tf

from absl.testing import parameterized

from tensorflow_model_optimization.python.core.clustering.keras import clusterable_layer
from tensorflow_model_optimization.python.core.clustering.keras import clustering_registry

keras = tf.keras
k = keras.backend
layers = keras.layers

errors_impl = tf.errors
test = tf.test

ClusterRegistry = clustering_registry.ClusteringRegistry
ClusteringLookupRegistry = clustering_registry.ClusteringLookupRegistry


class ClusteringAlgorithmTest(parameterized.TestCase):
  """Unit tests for clustering lookup algorithms"""

  def _pull_values(self, ca, pulling_indices, expected_output):
    pulling_indices_np = np.array(pulling_indices)
    res_tf = ca.get_clustered_weight(pulling_indices_np)

    res_np = k.batch_get_value([res_tf])[0]
    res_np_list = res_np.tolist()

    self.assertSequenceEqual(res_np_list, expected_output)

  @parameterized.parameters(
      ([-1, 1], [[0, 0, 1], [1, 1, 1]], [[-1, -1, 1], [1, 1, 1]]),
      ([-1, 0, 1], [[1, 1, 1], [1, 1, 1]], [[0, 0, 0], [0, 0, 0]]),
  )
  def testDenseWeightsCA(self,
                         clustering_centroids,
                         pulling_indices,
                         expected_output):
    """
    Verifies that DenseWeightsCA works as expected.
    """
    ca = clustering_registry.DenseWeightsCA(clustering_centroids)
    self._pull_values(ca, pulling_indices, expected_output)

  @parameterized.parameters(
      ([-1, 1], [0, 0, 0, 0, 1], [-1, -1, -1, -1, 1]),
      ([0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3]),
  )
  def testBiasWeightsCA(self,
                        clustering_centroids,
                        pulling_indices,
                        expected_output):
    """
    Verifies that BiasWeightsCA works as expected.
    """
    ca = clustering_registry.BiasWeightsCA(clustering_centroids)
    self._pull_values(ca, pulling_indices, expected_output)

  @parameterized.parameters(
      ([0, 3], [[[[0, 0, 0], [1, 1, 1], [0, 0, 0]]]],
       [[[[0, 0, 0], [3, 3, 3], [0, 0, 0]]]]),
      ([0, 3, 5], [[[[0, 1, 2], [1, 1, 1], [2, 1, 0]]]],
       [[[[0, 3, 5], [3, 3, 3], [5, 3, 0]]]]),
  )
  def testConvolutionalWeightsCA(self,
                                 clustering_centroids,
                                 pulling_indices,
                                 expected_output):
    """
    Verifies that ConvolutionalWeightsCA works as expected.
    """
    ca = clustering_registry.ConvolutionalWeightsCA(clustering_centroids)
    self._pull_values(ca, pulling_indices, expected_output)


class CustomLayer(layers.Layer):
  """A custom non-clusterable layer class."""


class ClusteringLookupRegistryTest(test.TestCase, parameterized.TestCase):
  """Unit tests for the ClusteringLookupRegistry class."""

  def testLookupHasEverythingFromRegistry(self):
    """
    Verifies that every layer that has non-empty ClusteringRegistry records is
    also presented in the ClusteringLookup.
    """
    for layer, clustering_record in ClusterRegistry._LAYERS_WEIGHTS_MAP.items():
      if clustering_record == []:
        continue

      self.assertIn(layer, ClusteringLookupRegistry._LAYERS_RESHAPE_MAP)

      for cr in clustering_record:
        self.assertIn(cr, ClusteringLookupRegistry._LAYERS_RESHAPE_MAP[layer])

  def testGetClusteringImplFailsWithUnknonwClassUnknownWeight(self):
    """
    Verifies that get_clustering_impl() raises an error when invoked with an
    unsupported layer class and an unsupported weight name.
    """
    with self.assertRaises(ValueError):
      ClusteringLookupRegistry.get_clustering_impl(CustomLayer(),
                                                   'no_such_weight')

  def testGetClusteringImplFailsWithKnonwClassUnknownWeight(self):
    """
    Verifies that get_clustering_impl() raises an error when invoked with a
    supported layer class and an unsupported weight name.
    """
    with self.assertRaises(ValueError):
      ClusteringLookupRegistry.get_clustering_impl(layers.Dense(10),
                                                   'no_such_weight')

  @parameterized.parameters(
      (layers.Conv2D, 'kernel', clustering_registry.ConvolutionalWeightsCA),
      (layers.Conv1D, 'kernel', clustering_registry.ConvolutionalWeightsCA),
  )
  def testReturnsResultsForKnownTypeKnownWeights(self,
                                                 layer_type,
                                                 weight,
                                                 expected):
    """
    Verifies that get_clustering_impl() returns the expected clustering lookup
    algorithm for the inputs provided.
    """
    # layer_type is a class, thus constructing an object here
    self.assertTrue(ClusteringLookupRegistry.get_clustering_impl(
        layer_type(32, 3), weight) is expected)

  def testRegisterNewImplWorks(self):
    """
    Verifies that registering a custom clustering lookup algorithm works as
    expected.
    """
    class NewKernelCA(clustering_registry.AbstractClusteringAlgorithm):

      def get_pulling_indices(self, weight):
        return 1, 2, 3

    new_impl = {
        CustomLayer: {
            'new_kernel': NewKernelCA
        }
    }

    ClusteringLookupRegistry.register_new_implementation(new_impl)
    self.assertTrue(ClusteringLookupRegistry.get_clustering_impl(
        CustomLayer(), 'new_kernel') is NewKernelCA)

  def testFailsIfNotADictIsGivenAsInput(self):
    """
    Verifies that registering a custom clustering lookup algorithm fails if the
    input provided is not a dict.
    """
    with self.assertRaises(TypeError):
      ClusteringLookupRegistry.register_new_implementation([1, 2, 3, 4])

  def testFailsIfNotADictIsGivenAsConcreteImplementation(self):
    """
    Verifies that registering a custom clustering lookup algorithm fails if the
    input provided for the concrete implementation is not a dict.
    """
    with self.assertRaises(TypeError):
      ClusteringLookupRegistry.register_new_implementation({
          ClusteringLookupRegistry: [('new_kernel', lambda x: x)]
      })


class ClusterRegistryTest(test.TestCase):
  """Unit tests for the ClusteringRegistry class."""

  class CustomLayerFromClusterableLayer(layers.Dense):
    """A custom layer class derived from a built-in clusterable layer."""
    pass

  class MinimalRNNCell(keras.layers.Layer):
    """A minimal RNN cell implementation."""

    def __init__(self, units, **kwargs):
      self.units = units
      self.state_size = units
      super(ClusterRegistryTest.MinimalRNNCell, self).__init__(**kwargs)

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
      h = k.dot(inputs, self.kernel)
      output = h + k.dot(prev_output, self.recurrent_kernel)
      return output, [output]

  class MinimalRNNCellClusterable(MinimalRNNCell,
                                  clusterable_layer.ClusterableLayer):
    """A clusterable minimal RNN cell implementation."""

    def get_clusterable_weights(self):
      return [
          ('kernel', self.kernel),
          ('recurrent_kernel', self.recurrent_kernel)
      ]

  def testSupportsKerasClusterableLayer(self):
    """
    Verifies that ClusterRegistry supports a built-in clusterable layer.
    """
    self.assertTrue(ClusterRegistry.supports(layers.Dense(10)))

  def testSupportsKerasClusterableLayerAlias(self):
    """
    Verifies that ClusterRegistry supports a built-in clusterable layer alias.
    """
    # layers.Conv2D maps to layers.convolutional.Conv2D
    self.assertTrue(ClusterRegistry.supports(layers.Conv2D(10, 5)))

  def testSupportsKerasNonClusterableLayer(self):
    """
    Verifies that ClusterRegistry supports a built-in non-clusterable layer.
    """
    # Dropout is a layer known to not be clusterable.
    self.assertTrue(ClusterRegistry.supports(layers.Dropout(0.5)))

  def testDoesNotSupportKerasUnsupportedLayer(self):
    """
    Verifies that ClusterRegistry does not support an unknown built-in layer.
    """
    # ConvLSTM2D is a built-in keras layer but not supported.
    self.assertFalse(ClusterRegistry.supports(layers.ConvLSTM2D(2, (5, 5))))

  def testSupportsKerasRNNLayers(self):
    """
    Verifies that ClusterRegistry supports the expected built-in RNN layers.
    """
    self.assertTrue(ClusterRegistry.supports(layers.LSTM(10)))
    self.assertTrue(ClusterRegistry.supports(layers.GRU(10)))
    self.assertTrue(ClusterRegistry.supports(layers.SimpleRNN(10)))

  def testDoesNotSupportKerasRNNLayerUnknownCell(self):
    """
    Verifies that ClusterRegistry does not support a custom non-clusterable RNN
    cell.
    """
    self.assertFalse(ClusterRegistry.supports(
        keras.layers.RNN(ClusterRegistryTest.MinimalRNNCell(32))))

  def testSupportsKerasRNNLayerClusterableCell(self):
    """
    Verifies that ClusterRegistry supports a custom clusterable RNN cell.
    """
    self.assertTrue(ClusterRegistry.supports(
        keras.layers.RNN(ClusterRegistryTest.MinimalRNNCellClusterable(32))))

  def testDoesNotSupportCustomLayer(self):
    """
    Verifies that ClusterRegistry does not support a custom non-clusterable
    layer.
    """
    self.assertFalse(ClusterRegistry.supports(CustomLayer(10)))

  def testDoesNotSupportCustomLayerInheritedFromClusterableLayer(self):
    """
    Verifies that ClusterRegistry does not support a custom layer derived from
    a clusterable layer.
    """
    self.assertFalse(
        ClusterRegistry.supports(
            ClusterRegistryTest.CustomLayerFromClusterableLayer(10)))

  def testMakeClusterableRaisesErrorForKerasUnsupportedLayer(self):
    """
    Verifies that an unsupported built-in layer cannot be made clusterable by
    calling make_clusterable().
    """
    with self.assertRaises(ValueError):
      ClusterRegistry.make_clusterable(layers.ConvLSTM2D(2, (5, 5)))

  def testMakeClusterableRaisesErrorForCustomLayer(self):
    """
    Verifies that a custom non-clusterable layer cannot be made clusterable by
    calling make_clusterable().
    """
    with self.assertRaises(ValueError):
      ClusterRegistry.make_clusterable(CustomLayer(10))

  def testMakeClusterableRaisesErrorForCustomLayerInheritedFromClusterableLayer(
      self):
    """
    Verifies that a non-clusterable layer derived from a clusterable layer
    cannot be made clusterable by calling make_clusterable().
    """
    with self.assertRaises(ValueError):
      ClusterRegistry.make_clusterable(
          ClusterRegistryTest.CustomLayerFromClusterableLayer(10))

  def testMakeClusterableWorksOnKerasClusterableLayer(self):
    """
    Verifies that make_clusterable() works as expected on a built-in
    clusterable layer.
    """
    layer = layers.Dense(10)
    with self.assertRaises(AttributeError):
      layer.get_clusterable_weights()

    ClusterRegistry.make_clusterable(layer)
    # Required since build method sets up the layer weights.
    keras.Sequential([layer]).build(input_shape=(10, 1))

    self.assertEqual([('kernel', layer.kernel)],
                     layer.get_clusterable_weights())

  def testMakeClusterableWorksOnKerasNonClusterableLayer(self):
    """
    Verifies that make_clusterable() works as expected on a built-in
    non-clusterable layer.
    """
    layer = layers.Dropout(0.5)
    with self.assertRaises(AttributeError):
      layer.get_clusterable_weights()

    ClusterRegistry.make_clusterable(layer)

    self.assertEqual([], layer.get_clusterable_weights())

  def testMakeClusterableWorksOnKerasRNNLayer(self):
    """
    Verifies that make_clusterable() works as expected on a built-in
    RNN layer.
    """
    layer = layers.LSTM(10)
    with self.assertRaises(AttributeError):
      layer.get_clusterable_weights()

    ClusterRegistry.make_clusterable(layer)
    keras.Sequential([layer]).build(input_shape=(2, 3, 4))

    expected_weights = [
        ('kernel', layer.cell.kernel),
        ('recurrent_kernel', layer.cell.recurrent_kernel)
    ]
    self.assertEqual(expected_weights, layer.get_clusterable_weights())

  def testMakeClusterableWorksOnKerasRNNLayerWithRNNCellsParams(self):
    """
    Verifies that make_clusterable() works as expected on a built-in
    RNN layer with built-in RNN cells.
    """
    cell1 = layers.LSTMCell(10)
    cell2 = layers.GRUCell(5)
    layer = layers.RNN([cell1, cell2])
    with self.assertRaises(AttributeError):
      layer.get_clusterable_weights()

    ClusterRegistry.make_clusterable(layer)
    keras.Sequential([layer]).build(input_shape=(2, 3, 4))

    expected_weights = [
        ('kernel', cell1.kernel),
        ('recurrent_kernel', cell1.recurrent_kernel),
        ('kernel', cell2.kernel),
        ('recurrent_kernel', cell2.recurrent_kernel)
    ]
    self.assertEqual(expected_weights, layer.get_clusterable_weights())

  def testMakeClusterableWorksOnKerasRNNLayerWithClusterableCell(self):
    """
    Verifies that make_clusterable() works as expected on a built-in
    RNN layer with a custom clusterable RNN cell.
    """
    cell1 = layers.LSTMCell(10)
    cell2 = ClusterRegistryTest.MinimalRNNCellClusterable(5)
    layer = layers.RNN([cell1, cell2])
    with self.assertRaises(AttributeError):
      layer.get_clusterable_weights()

    ClusterRegistry.make_clusterable(layer)
    keras.Sequential([layer]).build(input_shape=(2, 3, 4))

    expected_weights = [
        ('kernel', cell1.kernel),
        ('recurrent_kernel', cell1.recurrent_kernel),
        ('kernel', cell2.kernel),
        ('recurrent_kernel', cell2.recurrent_kernel)
    ]
    self.assertEqual(expected_weights, layer.get_clusterable_weights())

  def testMakeClusterableRaisesErrorOnRNNLayersUnsupportedCell(self):
    """
    Verifies that make_clusterable() raises an exception when invoked with a
    built-in RNN layer that contains a non-clusterable custom RNN cell.
    """
    with self.assertRaises(ValueError):
      ClusterRegistry.make_clusterable(layers.RNN(
          [layers.LSTMCell(10), ClusterRegistryTest.MinimalRNNCell(5)]))


if __name__ == '__main__':
  test.main()
