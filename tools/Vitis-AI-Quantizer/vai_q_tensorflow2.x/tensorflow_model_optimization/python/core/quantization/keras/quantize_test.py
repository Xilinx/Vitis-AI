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
"""Tests for quantize API functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_model_optimization.python.core.keras import test_utils as keras_test_utils
from tensorflow_model_optimization.python.core.quantization.keras import quantize
from tensorflow_model_optimization.python.core.quantization.keras import quantize_annotate as quantize_annotate_mod
from tensorflow_model_optimization.python.core.quantization.keras import quantize_config as quantize_config_mod
from tensorflow_model_optimization.python.core.quantization.keras import quantize_layer
from tensorflow_model_optimization.python.core.quantization.keras import quantize_wrapper as quantize_wrapper_mod
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry

quantize_annotate_layer = quantize.quantize_annotate_layer
quantize_annotate_model = quantize.quantize_annotate_model
quantize_apply = quantize.quantize_apply
QuantizeAnnotate = quantize_annotate_mod.QuantizeAnnotate
QuantizeWrapper = quantize_wrapper_mod.QuantizeWrapper

keras = tf.keras
K = tf.keras.backend
custom_object_scope = tf.keras.utils.custom_object_scope


class _TestQuantizeConfig(quantize_config_mod.QuantizeConfig):

  def get_weights_and_quantizers(self, layer):
    return []

  def get_activations_and_quantizers(self, layer):
    return []

  def set_quantize_weights(self, layer, quantize_weights):
    pass

  def set_quantize_activations(self, layer, quantize_activations):
    pass

  def get_output_quantizers(self, layer):
    pass

  def get_config(self):
    return {}


class QuantizeTest(tf.test.TestCase):

  def testQuantizeModel_Passes(self):
    model = keras.Sequential(
        [keras.layers.Dense(10, input_shape=(5,)),
         keras.layers.Dropout(0.4)])

    quantize.quantize_model(model)

  def testQuantizeLayer_Fails(self):
    layer = keras.layers.Dense(10, input_shape=(5,))

    with self.assertRaises(ValueError):
      quantize.quantize_model(layer)


class QuantizeAnnotateTest(tf.test.TestCase):

  def _assertWrappedLayer(self, layer, quantize_config=None):
    self.assertIsInstance(layer, quantize_annotate_mod.QuantizeAnnotate)
    self.assertEqual(quantize_config, layer.quantize_config)

  def _assertWrappedModel(self, model):
    for layer in model.layers:
      self._assertWrappedLayer(layer)

  def testQuantizeAnnotateLayer(self):
    layer = keras.layers.Dense(10, input_shape=(5,))
    wrapped_layer = quantize_annotate_layer(layer)

    self._assertWrappedLayer(wrapped_layer)

    inputs = np.random.rand(1, 5)
    model = keras.Sequential([layer])
    wrapped_model = keras.Sequential([wrapped_layer])

    # Both models should have the same results, since quantize_annotate does
    # not modify behavior.
    self.assertAllEqual(model.predict(inputs), wrapped_model.predict(inputs))

  def testQuantizeAnnotateSequentialFirstLayer_IsBuilt(self):
    model = keras.Sequential([
        quantize_annotate_layer(keras.layers.Dense(10, input_shape=(5,))),
        keras.layers.Dropout(0.4)
    ])

    self.assertTrue(model.built)

  def testQuantizeAnnotateLayer_FailsWithModel(self):
    model = keras_test_utils.build_simple_dense_model()

    with self.assertRaises(ValueError):
      quantize.quantize_annotate_layer(model)

  def testQuantizeAnnotateModel(self):
    model = keras.Sequential([
        keras.layers.Dense(10, input_shape=(5,)),
        keras.layers.Dropout(0.4)
    ])
    annotated_model = quantize_annotate_model(model)

    self._assertWrappedModel(annotated_model)

    inputs = np.random.rand(1, 5)
    self.assertAllEqual(model.predict(inputs), annotated_model.predict(inputs))

  def testQuantizeAnnotateModel_HasAnnotatedLayers(self):
    quantize_config = _TestQuantizeConfig()

    model = keras.Sequential([
        keras.layers.Dense(10, input_shape=(5,)),
        quantize_annotate_layer(
            keras.layers.Dense(5), quantize_config=quantize_config)
    ])
    annotated_model = quantize_annotate_model(model)

    self._assertWrappedLayer(annotated_model.layers[0])
    self._assertWrappedLayer(annotated_model.layers[1], quantize_config)
    # Ensure an already annotated layer is not wrapped again.
    self.assertIsInstance(annotated_model.layers[1].layer, keras.layers.Dense)

    inputs = np.random.rand(1, 5)
    self.assertAllEqual(model.predict(inputs), annotated_model.predict(inputs))

  def testQuantizeAnnotateModel_FailsWithLayer(self):
    layer = keras.layers.Dense(10)

    with self.assertRaises(ValueError):
      quantize.quantize_annotate_model(layer)

  class CustomLayer(keras.layers.Dense):
    pass

  def testQuantizeAnnotateModel_PassesWithCustomLayer(self):
    model = keras.Sequential([self.CustomLayer(3, input_shape=(2,))])
    quantize_annotate_model(model)

  # TODO(tfmot): this behavior may change in the future. If a user
  # start training a model without quantization and then wants to apply
  # it, not removing the optimizer would allow them to skip recompiling
  # the model.
  def testQuantizeAnnotateModel_RemovesOptimizer(self):
    model = keras_test_utils.build_simple_dense_model()
    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    self.assertIsNotNone(model.optimizer)

    annotated_model = quantize_annotate_model(model)
    self.assertIsNone(annotated_model.optimizer)

  def testQuantizeAnnotateModel_FailsWithSubclassedModel(self):
    class MyModel(keras.Model):
      def call(self, inputs, training=None, mask=None):  # pylint: disable=g-wrong-blank-lines
        return inputs

    with self.assertRaises(ValueError):
      quantize_annotate_model(MyModel())

  def testQuantizeAnnotateModel_FailsWithNestedModels(self):
    with self.assertRaises(ValueError):
      quantize_annotate_model(
          keras.Sequential(
              [keras.Sequential([keras.layers.Dense(10, input_shape=(2,))])]))


class QuantizeApplyTest(tf.test.TestCase):

  # Validation tests

  def testRaisesErrorIfNotKerasModel(self):
    with self.assertRaises(ValueError):
      quantize_apply(keras.layers.Dense(32))

  def testRaisesErrorIfKerasSubclassedModel(self):
    class MyModel(keras.Model):
      def call(self, inputs, training=None, mask=None):  # pylint: disable=g-wrong-blank-lines
        return inputs

    with self.assertRaises(ValueError):
      quantize_apply(MyModel())

  def testRaisesErrorNoAnnotatedLayers_Sequential(self):
    model = keras.Sequential([
        keras.layers.Dense(10), keras.layers.Dropout(0.4)])

    with self.assertRaises(ValueError):
      quantize_apply(model)

  def testRaisesErrorNoAnnotatedLayers_Functional(self):
    inputs = keras.Input(shape=(10,))
    x = keras.layers.Dense(32, activation='relu')(inputs)
    results = keras.layers.Dense(5, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=results)

    with self.assertRaises(ValueError):
      quantize_apply(model)

  def testRaisesErrorModelNotBuilt(self):
    model = keras.Sequential([quantize_annotate_layer(keras.layers.Dense(10))])

    self.assertFalse(model.built)
    with self.assertRaises(ValueError):
      quantize_apply(model)

  def testRaisesErrorNotInstanceOfQuantizeConfig(self):
    with self.assertRaises(ValueError):
      keras.Sequential([
          quantize_annotate_layer(
              keras.layers.Dense(10),
              quantize_config=object())
      ])

  # Helper functions to verify quantize wrapper applied correctly.

  def _assert_weights_equal_value(self, annotated_weights, emulated_weights):
    annotated_weight_values = K.batch_get_value(annotated_weights)
    emulated_weight_values = K.batch_get_value(emulated_weights)

    self.assertEqual(len(annotated_weight_values), len(emulated_weight_values))
    for aw, ew in zip(annotated_weight_values, emulated_weight_values):
      self.assertAllClose(aw, ew)

  def _assert_weights_different_objects(
      self, annotated_weights, emulated_weights):
    self.assertEqual(len(annotated_weights), len(emulated_weights))
    for aw, ew in zip(annotated_weights, emulated_weights):
      self.assertNotEqual(id(aw), id(ew))

  def _assert_layer_quantized(
      self, annotate_wrapper, quantize_wrapper, exclude_keys=None):
    self.assertIsInstance(quantize_wrapper, QuantizeWrapper)

    # Extract configs of the inner layers they wrap.
    annotated_config = annotate_wrapper.layer.get_config()
    quantized_config = quantize_wrapper.layer.get_config()

    # The underlying layers aren't always exactly the same. For example,
    # activations in the underlying layers might be replaced. Exclude keys
    # if required.
    if exclude_keys:
      for key in exclude_keys:
        annotated_config.pop(key)
        quantized_config.pop(key)

    self.assertEqual(annotated_config, quantized_config)

    def _sort_weights(weights):
      # Variables are named `quantize_annotate0/kernel:0` and
      # `quantize_emulate0/kernel:0`. Strip layer name to sort.
      return sorted(weights, key=lambda w: w.name.split('/')[1])

    annotated_weights = _sort_weights(annotate_wrapper.trainable_weights)
    quantized_weights = _sort_weights(quantize_wrapper.trainable_weights)

    # Quantized model should pick the same weight values from the original
    # model. However, they should not be the same weight objects. We don't
    # want training the quantized model to change weights in the original model.
    self._assert_weights_different_objects(annotated_weights, quantized_weights)
    self._assert_weights_equal_value(annotated_weights, quantized_weights)

  def _assert_model_quantized(
      self, annotated_model, quantized_model, exclude_keys=None):
    for layer_annotated, layer_quantized in \
        zip(annotated_model.layers, quantized_model.layers):

      if not isinstance(layer_annotated, QuantizeAnnotate):
        self.assertNotIsInstance(layer_quantized, QuantizeWrapper)
        continue

      self._assert_layer_quantized(
          layer_annotated, layer_quantized, exclude_keys)

  # quantize_apply Tests

  class CustomLayer(keras.layers.Dense):
    pass

  def testQuantizeCustomLayerWithoutQuantizeScope_RaisesError(self):
    annotated_model = keras.Sequential(
        [quantize_annotate_layer(self.CustomLayer(3, input_shape=(2,)))])

    with self.assertRaises(ValueError) as err:
      quantize_apply(annotated_model)

    expected_error = (
        'Unable to clone model. This generally happens if you used custom '
        'Keras layers or objects in your model. Please specify them via '
        '`quantize_scope` for your calls to `quantize_model` and '
        '`quantize_apply`.')

    self.assertEqual(str(err.exception), expected_error)

  def testQuantize_RaisesErrorIfNoQuantizeConfig(self):
    annotated_model = keras.Sequential([
        QuantizeAnnotate(self.CustomLayer(3), input_shape=(2,))])

    with custom_object_scope({'CustomLayer': self.CustomLayer}):
      with self.assertRaises(RuntimeError):
        quantize_apply(annotated_model)

  def testQuantize_UsesBuiltinQuantizeConfig(self):
    annotated_model = keras.Sequential([
        quantize_annotate_layer(keras.layers.Dense(3, input_shape=(2,)))])

    quantized_model = quantize_apply(annotated_model)
    quantized_layer = quantized_model.layers[1]

    # 'activation' gets replaced while quantizing the model. Hence excluded
    # from equality checks.
    self._assert_layer_quantized(
        annotated_model.layers[0], quantized_layer, ['activation'])
    self.assertIsInstance(
        quantized_layer.quantize_config,
        default_8bit_quantize_registry.Default8BitQuantizeConfig)

  def testQuantize_UsesQuantizeConfigFromUser_NoBuiltIn(self):
    annotated_model = keras.Sequential([
        quantize_annotate_layer(
            self.CustomLayer(3, input_shape=(2,)),
            quantize_config=_TestQuantizeConfig())
    ])

    with custom_object_scope({
        'CustomLayer': self.CustomLayer,
        '_TestQuantizeConfig': _TestQuantizeConfig
    }):
      quantized_model = quantize_apply(annotated_model)
    quantized_layer = quantized_model.layers[1]

    self._assert_layer_quantized(annotated_model.layers[0], quantized_layer)
    self.assertIsInstance(quantized_layer.quantize_config, _TestQuantizeConfig)

  def testQuantize_PreferenceToUserSpecifiedQuantizeConfig(self):
    annotated_model = keras.Sequential([
        quantize_annotate_layer(
            keras.layers.Dense(3, input_shape=(2,)),
            quantize_config=_TestQuantizeConfig())
    ])

    with custom_object_scope({'_TestQuantizeConfig': _TestQuantizeConfig}):
      quantized_model = quantize_apply(annotated_model)
    quantized_layer = quantized_model.layers[1]

    self._assert_layer_quantized(annotated_model.layers[0], quantized_layer)
    self.assertIsInstance(quantized_layer.quantize_config, _TestQuantizeConfig)

  def testAppliesQuantizationToAnnotatedModel_Sequential(self):
    model = keras.Sequential([
        keras.layers.Conv2D(32, 5, input_shape=(28, 28, 1), activation='relu'),
        quantize_annotate_layer(keras.layers.Dense(10, activation='relu')),
        quantize_annotate_layer(keras.layers.Dense(5, activation='softmax')),
    ])

    quantized_model = quantize_apply(model)

    self._assert_model_quantized(model, quantized_model, ['activation'])

  def testAppliesQuantizationToAnnotatedModel_PreservesBuiltState(self):
    model = keras_test_utils.build_simple_dense_model()
    annotated_model = quantize_annotate_model(model)

    self.assertTrue(annotated_model.built)

    quantized_model = quantize_apply(annotated_model)

    self.assertTrue(quantized_model.built)

  def _get_simple_functional_model(self):
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(32, 5, activation='relu')(inputs)
    x = quantize_annotate_layer(keras.layers.Dense(10, activation='relu'))(x)
    results = quantize_annotate_layer(
        keras.layers.Dense(5, activation='softmax'))(
            x)
    return keras.Model(inputs=inputs, outputs=results)

  def testAppliesQuantizationToAnnotatedModel_Functional(self):
    model = self._get_simple_functional_model()
    quantized_model = quantize_apply(model)

    self._assert_model_quantized(model, quantized_model, ['activation'])

  def testDoesNotQuantizeInputLayer_OutboundLayerNotQuantized(self):
    model = self._get_simple_functional_model()

    quantized_model = quantize_apply(model)

    # Since first layer is not quantized, QuantizeLayer does not get inserted
    # after InputLayer.

    input_layer = quantized_model._input_layers[0]
    next_layer = input_layer._outbound_nodes[0].outbound_layer
    self.assertNotIsInstance(next_layer, quantize_layer.QuantizeLayer)

  def testQuantizesInputLayer_OutboundLayerIsQuantized(self):
    inputs = keras.Input(shape=(28, 28, 1))
    x = quantize_annotate_layer(keras.layers.Conv2D(32, 5, activation='relu'))(
        inputs)
    x = quantize_annotate_layer(keras.layers.Dense(10, activation='relu'))(x)
    model = keras.Model(inputs=inputs, outputs=x)

    quantized_model = quantize_apply(model)

    # First layer is quantized. Hence QuantizeLayer gets inserted after
    # InputLayer.

    input_layer = quantized_model._input_layers[0]
    next_layer = input_layer._outbound_nodes[0].outbound_layer
    self.assertIsInstance(next_layer, quantize_layer.QuantizeLayer)

  # TODO(tfmot): this behavior may change in the future. If a user
  # start training a model without quantization and then wants to apply
  # it, not removing the optimizer would allow them to skip recompiling
  # the model.
  def testQuantizeApply_RemovesOptimizer(self):
    model = keras_test_utils.build_simple_dense_model()
    annotated_model = quantize_annotate_model(model)
    annotated_model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    self.assertIsNotNone(annotated_model.optimizer)

    quantized_model = quantize_apply(annotated_model)
    self.assertIsNone(quantized_model.optimizer)

  def testQuantizeApply_RunsWhenNestedModelNotAnnotated(self):
    annotated_model = keras.Sequential([
        keras.Sequential([keras.layers.Dense(10, input_shape=(2,))]),
        quantize_annotate_layer(keras.layers.Dense(10)),
    ])

    quantize_apply(annotated_model)


if __name__ == '__main__':
  tf.test.main()
