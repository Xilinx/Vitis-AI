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
"""Tests for keras clustering API."""

import json
import tensorflow as tf

from absl.testing import parameterized
from tensorflow.python.keras import keras_parameterized

from tensorflow_model_optimization.python.core.clustering.keras import cluster
from tensorflow_model_optimization.python.core.clustering.keras import cluster_config
from tensorflow_model_optimization.python.core.clustering.keras import cluster_wrapper
from tensorflow_model_optimization.python.core.clustering.keras import clusterable_layer
from tensorflow_model_optimization.python.core.clustering.keras import clustering_registry

keras = tf.keras
errors_impl = tf.errors
layers = keras.layers
test = tf.test


class TestModel(keras.Model):
  """A model subclass."""

  def __init__(self):
    """A test subclass model with one dense layer."""
    super(TestModel, self).__init__(name='test_model')
    self.layer1 = keras.layers.Dense(10, activation='relu')

  def call(self, inputs):
    return self.layer1(inputs)


class CustomClusterableLayer(layers.Dense, clusterable_layer.ClusterableLayer):

  def get_clusterable_weights(self):
    return [('kernel', self.kernel)]


class CustomNonClusterableLayer(layers.Dense):
  pass


class ClusterTest(test.TestCase, parameterized.TestCase):
  """Unit tests for the cluster module."""

  def setUp(self):
    super(ClusterTest, self).setUp()

    self.keras_clusterable_layer = layers.Dense(10)
    self.keras_non_clusterable_layer = layers.Dropout(0.4)
    self.keras_unsupported_layer = layers.ConvLSTM2D(2, (5, 5))  # Unsupported
    self.custom_clusterable_layer = CustomClusterableLayer(10)
    self.custom_non_clusterable_layer = CustomNonClusterableLayer(10)

    clustering_registry.ClusteringLookupRegistry.register_new_implementation(
        {
            CustomClusterableLayer: {
                'kernel': clustering_registry.DenseWeightsCA
            }
        }
    )

    self.model = keras.Sequential()
    self.params = {
        'number_of_clusters': 8,
        'cluster_centroids_init':
            cluster_config.CentroidInitialization.DENSITY_BASED
    }

  def _build_clustered_layer_model(self, layer):
    wrapped_layer = cluster.cluster_weights(layer, **self.params)
    self.model.add(wrapped_layer)
    self.model.build(input_shape=(10, 1))

    return wrapped_layer

  def _validate_clustered_layer(self, original_layer, wrapped_layer):
    self.assertIsInstance(wrapped_layer, cluster_wrapper.ClusterWeights)
    self.assertEqual(original_layer, wrapped_layer.layer)

  @staticmethod
  def _count_clustered_layers(model):
    count = 0
    for layer in model.layers:
      if isinstance(layer, cluster_wrapper.ClusterWeights):
        count += 1
    return count

  @keras_parameterized.run_all_keras_modes
  def testClusterKerasClusterableLayer(self):
    """
    Verifies that a built-in keras layer marked as clusterable is being
    clustered correctly.
    """
    wrapped_layer = self._build_clustered_layer_model(
        self.keras_clusterable_layer)

    self._validate_clustered_layer(self.keras_clusterable_layer, wrapped_layer)

  @keras_parameterized.run_all_keras_modes
  def testClusterKerasNonClusterableLayer(self):
    """
    Verifies that a built-in keras layer not marked as clusterable is
    not being clustered.
    """
    wrapped_layer = self._build_clustered_layer_model(
        self.keras_non_clusterable_layer)

    self._validate_clustered_layer(self.keras_non_clusterable_layer,
                                   wrapped_layer)
    self.assertEqual([], wrapped_layer.layer.get_clusterable_weights())

  def testClusterKerasUnsupportedLayer(self):
    """
    Verifies that attempting to cluster an unsupported layer raises an
    exception.
    """
    with self.assertRaises(ValueError):
      cluster.cluster_weights(self.keras_unsupported_layer, **self.params)

  @keras_parameterized.run_all_keras_modes
  def testClusterCustomClusterableLayer(self):
    """
    Verifies that a custom clusterable layer is being clustered correctly.
    """
    wrapped_layer = self._build_clustered_layer_model(
        self.custom_clusterable_layer)

    self._validate_clustered_layer(self.custom_clusterable_layer, wrapped_layer)
    self.assertEqual([('kernel', wrapped_layer.layer.kernel)],
                     wrapped_layer.layer.get_clusterable_weights())

  def testClusterCustomNonClusterableLayer(self):
    """
    Verifies that attempting to cluster a custom non-clusterable layer raises
    an exception.
    """
    with self.assertRaises(ValueError):
      cluster_wrapper.ClusterWeights(self.custom_non_clusterable_layer,
                                     **self.params)

  @keras_parameterized.run_all_keras_modes
  def testClusterSequentialModelSelectively(self):
    """
    Verifies that layers within a sequential model can be clustered
    selectively.
    """
    clustered_model = keras.Sequential()
    clustered_model.add(cluster.cluster_weights(self.keras_clusterable_layer, **self.params))
    clustered_model.add(self.keras_clusterable_layer)
    clustered_model.build(input_shape=(1, 10))

    self.assertIsInstance(clustered_model.layers[0], cluster_wrapper.ClusterWeights)
    self.assertNotIsInstance(clustered_model.layers[1], cluster_wrapper.ClusterWeights)

  @keras_parameterized.run_all_keras_modes
  def testClusterFunctionalModelSelectively(self):
    """
    Verifies that layers within a functional model can be clustered
    selectively.
    """
    i1 = keras.Input(shape=(10,))
    i2 = keras.Input(shape=(10,))
    x1 = cluster.cluster_weights(layers.Dense(10), **self.params)(i1)
    x2 = layers.Dense(10)(i2)
    outputs = layers.Add()([x1, x2])
    clustered_model = keras.Model(inputs=[i1, i2], outputs=outputs)

    self.assertIsInstance(clustered_model.layers[2], cluster_wrapper.ClusterWeights)
    self.assertNotIsInstance(clustered_model.layers[3], cluster_wrapper.ClusterWeights)

  @keras_parameterized.run_all_keras_modes
  def testClusterModelValidLayersSuccessful(self):
    """
    Verifies that clustering a sequential model results in all clusterable
    layers within the model being clustered.
    """
    model = keras.Sequential([
        self.keras_clusterable_layer,
        self.keras_non_clusterable_layer,
        self.custom_clusterable_layer
    ])
    clustered_model = cluster.cluster_weights(model, **self.params)
    clustered_model.build(input_shape=(1, 28, 28, 1))

    self.assertEqual(len(model.layers), len(clustered_model.layers))
    for layer, clustered_layer in zip(model.layers, clustered_model.layers):
      self._validate_clustered_layer(layer, clustered_layer)

  def testClusterModelUnsupportedKerasLayerRaisesError(self):
    """
    Verifies that attempting to cluster a model that contains an unsupported
    layer raises an exception.
    """
    with self.assertRaises(ValueError):
      cluster.cluster_weights(
          keras.Sequential([
              self.keras_clusterable_layer, self.keras_non_clusterable_layer,
              self.custom_clusterable_layer, self.keras_unsupported_layer
          ]), **self.params)

  def testClusterModelCustomNonClusterableLayerRaisesError(self):
    """
    Verifies that attempting to cluster a model that contains a custom
    non-clusterable layer raises an exception.
    """
    with self.assertRaises(ValueError):
      cluster.cluster_weights(
          keras.Sequential([
              self.keras_clusterable_layer, self.keras_non_clusterable_layer,
              self.custom_clusterable_layer, self.custom_non_clusterable_layer
          ]), **self.params)

  @keras_parameterized.run_all_keras_modes
  def testClusterModelDoesNotWrapAlreadyWrappedLayer(self):
    """
    Verifies that clustering a model that contains an already clustered layer
    does not result in wrapping the clustered layer into another
    cluster_wrapper.
    """
    model = keras.Sequential(
        [
            layers.Flatten(),
            cluster.cluster_weights(layers.Dense(10), **self.params),
        ])
    clustered_model = cluster.cluster_weights(model, **self.params)
    clustered_model.build(input_shape=(10, 10, 1))

    self.assertEqual(len(model.layers), len(clustered_model.layers))
    self._validate_clustered_layer(model.layers[0], clustered_model.layers[0])
    # Second layer is used as-is since it's already a clustered layer.
    self.assertEqual(model.layers[1], clustered_model.layers[1])
    self._validate_clustered_layer(model.layers[1].layer,
                                   clustered_model.layers[1])

  def testClusterValidLayersListSuccessful(self):
    """
    Verifies that clustering a list of layers results in all clusterable
    layers within the list being clustered.
    """
    model_layers = [
        self.keras_clusterable_layer,
        self.keras_non_clusterable_layer,
        self.custom_clusterable_layer
    ]
    clustered_list = cluster.cluster_weights(model_layers, **self.params)

    self.assertEqual(len(model_layers), len(clustered_list))
    for layer, clustered_layer in zip(model_layers, clustered_list):
      self._validate_clustered_layer(layer, clustered_layer)

  def testClusterSequentialModelNoInput(self):
    """
    Verifies that a sequential model without an input layer is being clustered
    correctly.
    """
    # No InputLayer
    model = keras.Sequential([
        layers.Dense(10),
        layers.Dense(10),
    ])
    clustered_model = cluster.cluster_weights(model, **self.params)
    self.assertEqual(self._count_clustered_layers(clustered_model), 2)

  @keras_parameterized.run_all_keras_modes
  def testClusterSequentialModelWithInput(self):
    """
    Verifies that a sequential model with an input layer is being clustered
    correctly.
    """
    # With InputLayer
    model = keras.Sequential([
        layers.Dense(10, input_shape=(10,)),
        layers.Dense(10),
    ])
    clustered_model = cluster.cluster_weights(model, **self.params)
    self.assertEqual(self._count_clustered_layers(clustered_model), 2)

  def testClusterSequentialModelPreservesBuiltStateNoInput(self):
    """
    Verifies that clustering a sequential model without an input layer
    preserves the built state of the model.
    """
    # No InputLayer
    model = keras.Sequential([
        layers.Dense(10),
        layers.Dense(10),
    ])
    self.assertEqual(model.built, False)
    clustered_model = cluster.cluster_weights(model, **self.params)
    self.assertEqual(model.built, False)

    # Test built state is preserved across serialization
    with cluster.cluster_scope():
      loaded_model = keras.models.model_from_config(
          json.loads(clustered_model.to_json()))
      self.assertEqual(loaded_model.built, False)

  @keras_parameterized.run_all_keras_modes
  def testClusterSequentialModelPreservesBuiltStateWithInput(self):
    """
    Verifies that clustering a sequential model with an input layer preserves
    the built state of the model.
    """
    # With InputLayer
    model = keras.Sequential([
        layers.Dense(10, input_shape=(10,)),
        layers.Dense(10),
    ])
    self.assertEqual(model.built, True)
    clustered_model = cluster.cluster_weights(model, **self.params)
    self.assertEqual(model.built, True)

    # Test built state is preserved across serialization
    with cluster.cluster_scope():
      loaded_model = keras.models.model_from_config(
          json.loads(clustered_model.to_json()))
    self.assertEqual(loaded_model.built, True)

  @keras_parameterized.run_all_keras_modes
  def testClusterFunctionalModelPreservesBuiltState(self):
    """
    Verifies that clustering a functional model preserves the built state of
    the model.
    """
    i1 = keras.Input(shape=(10,))
    i2 = keras.Input(shape=(10,))
    x1 = layers.Dense(10)(i1)
    x2 = layers.Dense(10)(i2)
    outputs = layers.Add()([x1, x2])
    model = keras.Model(inputs=[i1, i2], outputs=outputs)
    self.assertEqual(model.built, True)
    clustered_model = cluster.cluster_weights(model, **self.params)
    self.assertEqual(model.built, True)

    # Test built state preserves across serialization
    with cluster.cluster_scope():
      loaded_model = keras.models.model_from_config(
          json.loads(clustered_model.to_json()))
    self.assertEqual(loaded_model.built, True)

  @keras_parameterized.run_all_keras_modes
  def testClusterFunctionalModel(self):
    """
    Verifies that a functional model is being clustered correctly.
    """
    i1 = keras.Input(shape=(10,))
    i2 = keras.Input(shape=(10,))
    x1 = layers.Dense(10)(i1)
    x2 = layers.Dense(10)(i2)
    outputs = layers.Add()([x1, x2])
    model = keras.Model(inputs=[i1, i2], outputs=outputs)
    clustered_model = cluster.cluster_weights(model, **self.params)
    self.assertEqual(self._count_clustered_layers(clustered_model), 3)

  @keras_parameterized.run_all_keras_modes
  def testClusterFunctionalModelWithLayerReused(self):
    """
    Verifies that a layer reused within a functional model multiple times is
    only being clustered once.
    """
    # The model reuses the Dense() layer. Make sure it's only clustered once.
    inp = keras.Input(shape=(10,))
    dense_layer = layers.Dense(10)
    x = dense_layer(inp)
    x = dense_layer(x)
    model = keras.Model(inputs=[inp], outputs=[x])
    clustered_model = cluster.cluster_weights(model, **self.params)
    self.assertEqual(self._count_clustered_layers(clustered_model), 1)

  @keras_parameterized.run_all_keras_modes
  def testClusterSubclassModel(self):
    """
    Verifies that attempting to cluster an instance of a subclass of
    keras.Model raises an exception.
    """
    model = TestModel()
    with self.assertRaises(ValueError):
      _ = cluster.cluster_weights(model, **self.params)

  @keras_parameterized.run_all_keras_modes
  def testStripClusteringSequentialModel(self):
    """
    Verifies that stripping the clustering wrappers from a sequential model
    produces the expected config.
    """
    model = keras.Sequential([
        layers.Dense(10),
        layers.Dense(10),
    ])

    clustered_model = cluster.cluster_weights(model, **self.params)
    stripped_model = cluster.strip_clustering(clustered_model)

    self.assertEqual(self._count_clustered_layers(stripped_model), 0)
    self.assertEqual(model.get_config(), stripped_model.get_config())

  @keras_parameterized.run_all_keras_modes
  def testClusterStrippingFunctionalModel(self):
    """
    Verifies that stripping the clustering wrappers from a functional model
    produces the expected config.
    """
    i1 = keras.Input(shape=(10,))
    i2 = keras.Input(shape=(10,))
    x1 = layers.Dense(10)(i1)
    x2 = layers.Dense(10)(i2)
    outputs = layers.Add()([x1, x2])
    model = keras.Model(inputs=[i1, i2], outputs=outputs)

    clustered_model = cluster.cluster_weights(model, **self.params)
    stripped_model = cluster.strip_clustering(clustered_model)

    self.assertEqual(self._count_clustered_layers(stripped_model), 0)
    self.assertEqual(model.get_config(), stripped_model.get_config())

  @keras_parameterized.run_all_keras_modes
  def testClusterWeightsStrippedWeights(self):
    """
    Verifies that stripping the clustering wrappers from a functional model
    preserves the clustered weights.
    """
    i1 = keras.Input(shape=(10,))
    x1 = layers.BatchNormalization()(i1)
    outputs = x1
    model = keras.Model(inputs=[i1], outputs=outputs)

    clustered_model = cluster.cluster_weights(model, **self.params)
    cluster_weight_length = (len(clustered_model.get_weights()))
    stripped_model = cluster.strip_clustering(clustered_model)

    self.assertEqual(self._count_clustered_layers(stripped_model), 0)
    self.assertEqual(len(stripped_model.get_weights()), cluster_weight_length)

  @keras_parameterized.run_all_keras_modes
  def testStrippedKernel(self):
    """
    Verifies that stripping the clustering wrappers from a functional model
    restores the layers kernel and the layers weight array to the new clustered weight value .
    """
    i1 = keras.Input(shape=(1, 1, 1))
    x1 = layers.Conv2D(1, 1)(i1)
    outputs = x1
    model = keras.Model(inputs=[i1], outputs=outputs)

    clustered_model = cluster.cluster_weights(model, **self.params)
    clustered_conv2d_layer = clustered_model.layers[1]
    clustered_kernel = clustered_conv2d_layer.layer.kernel
    stripped_model = cluster.strip_clustering(clustered_model)
    stripped_conv2d_layer = stripped_model.layers[1]

    self.assertEqual(self._count_clustered_layers(stripped_model), 0)
    self.assertIsNot(stripped_conv2d_layer.kernel, clustered_kernel)
    self.assertEqual(stripped_conv2d_layer.kernel,
                     stripped_conv2d_layer.weights[0])

  @keras_parameterized.run_all_keras_modes
  def testStripSelectivelyClusteredFunctionalModel(self):
    """
    Verifies that invoking strip_clustering() on a selectively clustered
    functional model strips the clustering wrappers from the clustered layers.
    """
    i1 = keras.Input(shape=(10,))
    i2 = keras.Input(shape=(10,))
    x1 = cluster.cluster_weights(layers.Dense(10), **self.params)(i1)
    x2 = layers.Dense(10)(i2)
    outputs = layers.Add()([x1, x2])
    clustered_model = keras.Model(inputs=[i1, i2], outputs=outputs)

    stripped_model = cluster.strip_clustering(clustered_model)

    self.assertEqual(self._count_clustered_layers(stripped_model), 0)
    self.assertIsInstance(stripped_model.layers[2], layers.Dense)

  @keras_parameterized.run_all_keras_modes
  def testStripSelectivelyClusteredSequentialModel(self):
    """
    Verifies that invoking strip_clustering() on a selectively clustered
    sequential model strips the clustering wrappers from the clustered layers.
    """
    clustered_model = keras.Sequential([
      cluster.cluster_weights(layers.Dense(10), **self.params),
      layers.Dense(10),
    ])
    clustered_model.build(input_shape=(1, 10))

    stripped_model = cluster.strip_clustering(clustered_model)

    self.assertEqual(self._count_clustered_layers(stripped_model), 0)
    self.assertIsInstance(stripped_model.layers[0], layers.Dense)

if __name__ == '__main__':
  test.main()
