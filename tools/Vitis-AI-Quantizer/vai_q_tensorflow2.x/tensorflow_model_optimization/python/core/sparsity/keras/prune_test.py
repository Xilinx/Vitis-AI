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
"""Tests for tf.keras pruning APIs under prune.py."""

import json
import tempfile

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

# TODO(b/139939526): move to public API.
from tensorflow.python.keras import keras_parameterized
from tensorflow_model_optimization.python.core.keras import test_utils as keras_test_utils
from tensorflow_model_optimization.python.core.sparsity.keras import prunable_layer
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

keras = tf.keras
errors_impl = tf.errors
layers = keras.layers
test = tf.test


class TestSubclassedModel(keras.Model):
  """A model subclass."""

  def __init__(self):
    """A test subclass model with one dense layer."""
    super(TestSubclassedModel, self).__init__(name='test_model')
    self.layer1 = keras.layers.Dense(10, activation='relu')

  def call(self, inputs):
    return self.layer1(inputs)


class CustomPrunableLayer(layers.Dense, prunable_layer.PrunableLayer):

  def get_prunable_weights(self):
    return [self.kernel]


class CustomNonPrunableLayer(layers.Dense):
  pass


class PruneTest(test.TestCase, parameterized.TestCase):

  INVALID_TO_PRUNE_PARAM_ERROR = ('`prune_low_magnitude` can only prune an '
                                  'object of the following types: '
                                  'tf.keras.models.Sequential, tf.keras '
                                  'functional model, tf.keras.layers.Layer, '
                                  'list of tf.keras.layers.Layer. You passed an'
                                  ' object of type: {input}.')

  def setUp(self):
    super(PruneTest, self).setUp()

    # Layers passed in for Pruning can either be standard Keras layers provided
    # by the tf.keras API (these fall under the `keras.layers` namespace), or
    # custom layers provided by the user which inherit the base
    # `keras.layers.Layer`.
    # Standard Keras layers can either be Prunable (we know how to prune them),
    # Non-Prunable (we know these layers can't be pruned) and Unsupported (we
    # don't know how to deal with these yet.). Unsupported layers will raise an
    # error when tried to prune.
    # Custom Layers can either be Prunable (ie., they implement the
    # `PrunableLayer` interface, or Non-Prunable (they don't expose any pruning
    # behavior.)

    # TODO(pulkitb): Change naming of Prunable/Non-Prunable/Unsupported to be
    # clearer.
    self.keras_prunable_layer = layers.Dense(10)  # Supports pruning
    self.keras_non_prunable_layer = layers.Dropout(
        0.4)  # Pruning not applicable
    self.keras_unsupported_layer = layers.ConvLSTM2D(2, (5, 5))  # Unsupported
    self.custom_prunable_layer = CustomPrunableLayer(10)
    self.custom_non_prunable_layer = CustomNonPrunableLayer(10)

    self.model = keras.Sequential()
    self.params = {
        'pruning_schedule': pruning_schedule.ConstantSparsity(0.5, 0),
        'block_size': (1, 1),
        'block_pooling_type': 'AVG'
    }

  def _build_pruned_layer_model(self, layer):
    wrapped_layer = prune.prune_low_magnitude(layer, **self.params)
    self.model.add(wrapped_layer)
    self.model.build(input_shape=(10, 1))

    return wrapped_layer

  def _validate_pruned_layer(self, original_layer, wrapped_layer):
    self.assertIsInstance(wrapped_layer, pruning_wrapper.PruneLowMagnitude)
    self.assertEqual(original_layer, wrapped_layer.layer)
    self.assertEqual(
        self.params['pruning_schedule'], wrapped_layer.pruning_schedule)
    self.assertEqual(self.params['block_size'], wrapped_layer.block_size)
    self.assertEqual(
        self.params['block_pooling_type'], wrapped_layer.block_pooling_type)

  @staticmethod
  def _count_pruned_layers(model):
    count = 0
    for layer in model._layers:
      if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
        count += 1
    return count

  def testPruneKerasPrunableLayer(self):
    wrapped_layer = self._build_pruned_layer_model(self.keras_prunable_layer)

    self._validate_pruned_layer(self.keras_prunable_layer, wrapped_layer)

  def testPruneKerasNonPrunableLayer(self):
    wrapped_layer = self._build_pruned_layer_model(
        self.keras_non_prunable_layer)

    self._validate_pruned_layer(self.keras_non_prunable_layer, wrapped_layer)
    self.assertEqual([], wrapped_layer.layer.get_prunable_weights())

  def testPruneKerasUnsupportedLayer(self):
    with self.assertRaises(ValueError):
      prune.prune_low_magnitude(self.keras_unsupported_layer, **self.params)

  def testPruneCustomPrunableLayer(self):
    wrapped_layer = self._build_pruned_layer_model(self.custom_prunable_layer)

    self._validate_pruned_layer(self.custom_prunable_layer, wrapped_layer)
    self.assertEqual([wrapped_layer.layer.kernel],
                     wrapped_layer.layer.get_prunable_weights())

  def testPruneCustomNonPrunableLayer(self):
    with self.assertRaises(ValueError):
      prune.prune_low_magnitude(self.custom_non_prunable_layer, **self.params)

  def testPruneModelValidLayersSuccessful(self):
    model = keras.Sequential([
        self.keras_prunable_layer,
        self.keras_non_prunable_layer,
        self.custom_prunable_layer
    ])
    pruned_model = prune.prune_low_magnitude(model, **self.params)
    pruned_model.build(input_shape=(1, 28, 28, 1))

    self.assertEqual(len(model.layers), len(pruned_model.layers))
    for layer, pruned_layer in zip(model.layers, pruned_model.layers):
      self._validate_pruned_layer(layer, pruned_layer)

  def testPruneModelUnsupportedKerasLayerRaisesError(self):
    with self.assertRaises(ValueError):
      prune.prune_low_magnitude(
          keras.Sequential([
              self.keras_prunable_layer, self.keras_non_prunable_layer,
              self.custom_prunable_layer, self.keras_unsupported_layer
          ]), **self.params)

  def testPruneModelCustomNonPrunableLayerRaisesError(self):
    with self.assertRaises(ValueError):
      prune.prune_low_magnitude(
          keras.Sequential([
              self.keras_prunable_layer, self.keras_non_prunable_layer,
              self.custom_prunable_layer, self.custom_non_prunable_layer
          ]), **self.params)

  def testPruneModelDoesNotWrapAlreadyWrappedLayer(self):
    model = keras.Sequential(
        [layers.Dense(10),
         prune.prune_low_magnitude(layers.Dense(10), **self.params)])
    pruned_model = prune.prune_low_magnitude(model, **self.params)
    pruned_model.build(input_shape=(10, 1))

    self.assertEqual(len(model.layers), len(pruned_model.layers))
    self._validate_pruned_layer(model.layers[0], pruned_model.layers[0])
    # Second layer is used as-is since it's already a pruned layer.
    self.assertEqual(model.layers[1], pruned_model.layers[1])

  def testPruneValidLayersListSuccessful(self):
    model_layers = [
        self.keras_prunable_layer,
        self.keras_non_prunable_layer,
        self.custom_prunable_layer
    ]
    pruned_layers = prune.prune_low_magnitude(model_layers, **self.params)

    self.assertEqual(len(model_layers), len(pruned_layers))
    for layer, pruned_layer in zip(model_layers, pruned_layers):
      self._validate_pruned_layer(layer, pruned_layer)

  @keras_parameterized.run_all_keras_modes
  def testPruneInferenceWorks_PruningStepCallbackNotRequired(self):
    model = prune.prune_low_magnitude(
        keras.Sequential([
            layers.Dense(10, activation='relu', input_shape=(100,)),
            layers.Dense(2, activation='sigmoid')
        ]), **self.params)

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.SGD(),
        metrics=['accuracy'])

    model.predict(np.random.rand(1000, 100))
    model.evaluate(
        np.random.rand(1000, 100),
        keras.utils.to_categorical(np.random.randint(2, size=(1000, 1))))

  def testPruneSequentialModel(self):
    # No InputLayer
    model = keras.Sequential([
        layers.Dense(10),
        layers.Dense(10),
    ])
    pruned_model = prune.prune_low_magnitude(model, **self.params)
    self.assertEqual(self._count_pruned_layers(pruned_model), 2)

    # With InputLayer
    model = keras.Sequential([
        layers.Dense(10, input_shape=(10,)),
        layers.Dense(10),
    ])
    pruned_model = prune.prune_low_magnitude(model, **self.params)
    self.assertEqual(self._count_pruned_layers(pruned_model), 2)

  def testPruneSequentialModelPreservesBuiltState(self):
    # No InputLayer
    model = keras.Sequential([
        layers.Dense(10),
        layers.Dense(10),
    ])
    self.assertEqual(model.built, False)
    pruned_model = prune.prune_low_magnitude(model, **self.params)
    self.assertEqual(model.built, False)

    # Test built state preserves across serialization
    with prune.prune_scope():
      loaded_model = keras.models.model_from_config(
          json.loads(pruned_model.to_json()))
      self.assertEqual(loaded_model.built, False)

    # With InputLayer
    model = keras.Sequential([
        layers.Dense(10, input_shape=(10,)),
        layers.Dense(10),
    ])
    self.assertEqual(model.built, True)
    pruned_model = prune.prune_low_magnitude(model, **self.params)
    self.assertEqual(model.built, True)

    # Test built state preserves across serialization
    with prune.prune_scope():
      loaded_model = keras.models.model_from_config(
          json.loads(pruned_model.to_json()))
    self.assertEqual(loaded_model.built, True)

  def testPruneFunctionalModel(self):
    i1 = keras.Input(shape=(10,))
    i2 = keras.Input(shape=(10,))
    x1 = layers.Dense(10)(i1)
    x2 = layers.Dense(10)(i2)
    outputs = layers.Add()([x1, x2])
    model = keras.Model(inputs=[i1, i2], outputs=outputs)
    pruned_model = prune.prune_low_magnitude(model, **self.params)
    self.assertEqual(self._count_pruned_layers(pruned_model), 3)

  def testPruneFunctionalModelWithLayerReused(self):
    # The model reuses the Dense() layer. Make sure it's only pruned once.
    inp = keras.Input(shape=(10,))
    dense_layer = layers.Dense(10)
    x = dense_layer(inp)
    x = dense_layer(x)
    model = keras.Model(inputs=[inp], outputs=[x])
    pruned_model = prune.prune_low_magnitude(model, **self.params)
    self.assertEqual(self._count_pruned_layers(pruned_model), 1)

  def testPruneFunctionalModelPreservesBuiltState(self):
    i1 = keras.Input(shape=(10,))
    i2 = keras.Input(shape=(10,))
    x1 = layers.Dense(10)(i1)
    x2 = layers.Dense(10)(i2)
    outputs = layers.Add()([x1, x2])
    model = keras.Model(inputs=[i1, i2], outputs=outputs)
    self.assertEqual(model.built, True)
    pruned_model = prune.prune_low_magnitude(model, **self.params)
    self.assertEqual(model.built, True)

    # Test built state preserves across serialization
    with prune.prune_scope():
      loaded_model = keras.models.model_from_config(
          json.loads(pruned_model.to_json()))
    self.assertEqual(loaded_model.built, True)

  def testPruneSubclassModel(self):
    model = TestSubclassedModel()
    with self.assertRaises(ValueError) as e:
      _ = prune.prune_low_magnitude(model, **self.params)
    self.assertEqual(
        str(e.exception),
        self.INVALID_TO_PRUNE_PARAM_ERROR.format(input='TestSubclassedModel'))

  def testPruneMiscObject(self):

    model = object()
    with self.assertRaises(ValueError) as e:
      _ = prune.prune_low_magnitude(model, **self.params)
    self.assertEqual(
        str(e.exception),
        self.INVALID_TO_PRUNE_PARAM_ERROR.format(input='object'))

  def testStripPruningSequentialModel(self):
    model = keras.Sequential([
        layers.Dense(10),
        layers.Dense(10),
    ])

    pruned_model = prune.prune_low_magnitude(model, **self.params)
    stripped_model = prune.strip_pruning(pruned_model)
    self.assertEqual(self._count_pruned_layers(stripped_model), 0)
    self.assertEqual(model.get_config(), stripped_model.get_config())

  def testStripPruningFunctionalModel(self):
    i1 = keras.Input(shape=(10,))
    i2 = keras.Input(shape=(10,))
    x1 = layers.Dense(10)(i1)
    x2 = layers.Dense(10)(i2)
    outputs = layers.Add()([x1, x2])
    model = keras.Model(inputs=[i1, i2], outputs=outputs)

    pruned_model = prune.prune_low_magnitude(model, **self.params)
    stripped_model = prune.strip_pruning(pruned_model)

    self.assertEqual(self._count_pruned_layers(stripped_model), 0)
    self.assertEqual(model.get_config(), stripped_model.get_config())

  def testPruneScope_NeededForKerasModel(self):
    model = keras_test_utils.build_simple_dense_model()
    pruned_model = prune.prune_low_magnitude(model)

    _, keras_model = tempfile.mkstemp('.h5')
    pruned_model.save(keras_model)

    with self.assertRaises(ValueError):
      tf.keras.models.load_model(keras_model)

    # works with `prune_scope`
    with prune.prune_scope():
      tf.keras.models.load_model(keras_model)

  def testPruneScope_NotNeededForKerasCheckpoint(self):
    model = keras_test_utils.build_simple_dense_model()
    pruned_model = prune.prune_low_magnitude(model)

    _, keras_weights = tempfile.mkstemp('.h5')
    pruned_model.save_weights(keras_weights)

    same_architecture_model = keras_test_utils.build_simple_dense_model()
    same_architecture_model = prune.prune_low_magnitude(same_architecture_model)

    # would error if `prune_scope` was needed.
    same_architecture_model.load_weights(keras_weights)

  def testPruneScope_NotNeededForTFCheckpoint(self):
    model = keras_test_utils.build_simple_dense_model()
    pruned_model = prune.prune_low_magnitude(model)

    _, tf_weights = tempfile.mkstemp('.tf')
    pruned_model.save_weights(tf_weights)

    same_architecture_model = keras_test_utils.build_simple_dense_model()
    same_architecture_model = prune.prune_low_magnitude(same_architecture_model)

    # would error if `prune_scope` was needed.
    same_architecture_model.load_weights(tf_weights)

  def testPruneScope_NotNeededForTF2SavedModel(self):
    # TODO(tfmot): replace with shared v1 test_util.
    is_v1_apis = hasattr(tf, 'assign')
    if is_v1_apis:
      return

    model = keras_test_utils.build_simple_dense_model()
    pruned_model = prune.prune_low_magnitude(model)

    saved_model_dir = tempfile.mkdtemp()

    tf.saved_model.save(pruned_model, saved_model_dir)

    # would error if `prune_scope` was needed.
    tf.saved_model.load(saved_model_dir)

  def testPruneScope_NeededForTF1SavedModel(self):
    # TODO(tfmot): replace with shared v1 test_util.
    is_v1_apis = hasattr(tf, 'assign')
    if not is_v1_apis:
      return

    model = keras_test_utils.build_simple_dense_model()
    pruned_model = prune.prune_low_magnitude(model)

    saved_model_dir = tempfile.mkdtemp()

    tf.keras.experimental.export_saved_model(pruned_model, saved_model_dir)
    with self.assertRaises(ValueError):
      tf.keras.experimental.load_from_saved_model(saved_model_dir)

    # works with `prune_scope`
    with prune.prune_scope():
      tf.keras.experimental.load_from_saved_model(saved_model_dir)


if __name__ == '__main__':
  test.main()
