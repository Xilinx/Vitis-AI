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
"""Integration test which ensures user facing code paths work."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

# TODO(b/139939526): move to public API.
from tensorflow.python.keras import keras_parameterized

from tensorflow_model_optimization.python.core.keras import compat
from tensorflow_model_optimization.python.core.keras import test_utils
from tensorflow_model_optimization.python.core.quantization.keras import quantize
from tensorflow_model_optimization.python.core.quantization.keras import quantize_config
from tensorflow_model_optimization.python.core.quantization.keras import quantizers
from tensorflow_model_optimization.python.core.quantization.keras import utils

QuantizeConfig = quantize_config.QuantizeConfig
Quantizer = quantizers.Quantizer
MovingAverageQuantizer = quantizers.MovingAverageQuantizer

l = tf.keras.layers


# TODO(tfmot): enable for v1. Currently fails because the decorator
# on graph mode wraps everything in a graph, which is not compatible
# with the TFLite converter's call to clear_session().
@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class QuantizeIntegrationTest(tf.test.TestCase, parameterized.TestCase):

  @staticmethod
  def _batch(dims, batch_size):
    """Adds provided batch_size to existing dims.

    If dims is (None, 5, 2), returns (batch_size, 5, 2)

    Args:
      dims: Dimensions
      batch_size: batch_size

    Returns:
      dims with batch_size added as first parameter of list.
    """
    if dims[0] is None:
      dims[0] = batch_size
    return dims

  def _assert_models_equal(self, model1, model2):
    model1_config = model1.get_config()
    model1_config.pop('build_input_shape', None)
    model2_config = model2.get_config()
    model2_config.pop('build_input_shape', None)
    self.assertEqual(model1_config, model2_config)
    self.assertAllClose(model1.get_weights(), model2.get_weights())

    self._assert_outputs_equal(model1, model2)

  # After saving a model to SavedModel and then loading it back,
  # the class changes, which results in config differences. This
  # may change after a sync (TF 2.2.0): TODO(alanchiao): try it.
  def _assert_outputs_equal(self, model1, model2):
    inputs = np.random.randn(
        *self._batch(model1.input.get_shape().as_list(), 1))
    self.assertAllClose(model1.predict(inputs), model2.predict(inputs))

  # TODO(tfmot): use shared test util that is model-independent.
  @staticmethod
  def _train_model(model):
    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(
        np.random.rand(20, 10),
        tf.keras.utils.to_categorical(np.random.randint(5, size=(20, 1)), 5),
        batch_size=20)

  ####################################################################
  # Tests for research with quantization.

  # Test example in quantization comprehensive guide
  class _FixedRangeQuantizer(Quantizer):
    """Quantizer which keeps values between -1 and 1."""

    # Test build function that returns no weights.
    def build(self, tensor_shape, name, layer):
      return {}

    def __call__(self, inputs, training, weights, **kwargs):
      return tf.keras.backend.clip(inputs, -1.0, 1.0)

    def get_config(self):
      return {}

  @staticmethod
  def _get_quant_params(quantizer_class):
    if quantizer_class == quantizers.LastValueQuantizer:
      return {
          'num_bits': 8,
          'per_axis': False,
          'symmetric': False,
          'narrow_range': False
      }
    else:
      return {}

  @parameterized.parameters(quantizers.LastValueQuantizer, _FixedRangeQuantizer)
  def testCustomWeightQuantizers_Run(self, quantizer_type):
    init_params = self._get_quant_params(quantizer_type)

    # Additional test that same quantizer object can be shared
    # between Configs, though we don't expicitly promote this
    # anywhere in the documentation.
    quantizer = quantizer_type(**init_params)

    class DenseQuantizeConfig(QuantizeConfig):
      """Custom QuantizeConfig for Dense layer."""

      def get_weights_and_quantizers(self, layer):
        return [(layer.kernel, quantizer)]

      def get_activations_and_quantizers(self, layer):
        # Defaults.
        return [(layer.activation,
                 MovingAverageQuantizer(
                     num_bits=8,
                     per_axis=False,
                     symmetric=False,
                     narrow_range=False))]

      def set_quantize_weights(self, layer, quantize_weights):
        layer.kernel = quantize_weights[0]

      def set_quantize_activations(self, layer, quantize_activations):
        return

      def get_output_quantizers(self, layer):
        return []

      def get_config(self):
        return {}

    annotated_model = tf.keras.Sequential([
        quantize.quantize_annotate_layer(
            l.Dense(8, input_shape=(10,)), DenseQuantizeConfig()),
        quantize.quantize_annotate_layer(
            l.Dense(5), DenseQuantizeConfig())
    ])

    with quantize.quantize_scope(
        {'DenseQuantizeConfig': DenseQuantizeConfig}):
      quant_model = quantize.quantize_apply(annotated_model)

    # Check no error happens.
    self._train_model(quant_model)

  ####################################################################
  # Tests for training with quantization with checkpointing.

  # TODO(pulkitb): Parameterize and add more model/runtime options.
  def testSerialization_KerasModel(self):
    model = test_utils.build_simple_dense_model()
    quantized_model = quantize.quantize_model(model)
    self._train_model(quantized_model)

    _, model_file = tempfile.mkstemp('.h5')
    tf.keras.models.save_model(quantized_model, model_file)
    with quantize.quantize_scope():
      loaded_model = tf.keras.models.load_model(model_file)

    self._assert_models_equal(quantized_model, loaded_model)

  def testSerialization_KerasCheckpoint(self):
    model = test_utils.build_simple_dense_model()
    quantized_model = quantize.quantize_model(model)
    self._train_model(quantized_model)

    _, keras_weights = tempfile.mkstemp('.h5')
    quantized_model.save_weights(keras_weights)

    same_architecture_model = test_utils.build_simple_dense_model()
    same_architecture_model = quantize.quantize_model(same_architecture_model)
    same_architecture_model.load_weights(keras_weights)

    self._assert_outputs_equal(quantized_model, same_architecture_model)

  def testSerialization_TF2SavedModel(self):
    if compat.is_v1_apis():
      return

    model = test_utils.build_simple_dense_model()
    quantized_model = quantize.quantize_model(model)
    self._train_model(quantized_model)

    model_dir = tempfile.mkdtemp()
    tf.keras.models.save_model(quantized_model, model_dir)
    loaded_model = tf.keras.models.load_model(model_dir)

    self._assert_outputs_equal(quantized_model, loaded_model)

  def testSerialization_TF1SavedModel(self):
    if not compat.is_v1_apis():
      return

    model = test_utils.build_simple_dense_model()
    quantized_model = quantize.quantize_model(model)
    self._train_model(quantized_model)

    saved_model_dir = tempfile.mkdtemp()
    with quantize.quantize_scope():
      tf.keras.experimental.export_saved_model(quantized_model, saved_model_dir)

    with quantize.quantize_scope():
      loaded_model = tf.keras.experimental.load_from_saved_model(
          saved_model_dir)

    self._assert_outputs_equal(quantized_model, loaded_model)

  def testSerialization_TFCheckpoint(self):
    model = test_utils.build_simple_dense_model()
    quantized_model = quantize.quantize_model(model)
    self._train_model(quantized_model)

    _, tf_weights = tempfile.mkstemp('.tf')
    quantized_model.save_weights(tf_weights)

    same_architecture_model = test_utils.build_simple_dense_model()
    same_architecture_model = quantize.quantize_model(same_architecture_model)
    same_architecture_model.load_weights(tf_weights)

    self._assert_outputs_equal(quantized_model, same_architecture_model)


if __name__ == '__main__':
  tf.test.main()
