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
"""Tests for Default Transforms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantize
from tensorflow_model_optimization.python.core.quantization.keras import quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras import quantize_layer
from tensorflow_model_optimization.python.core.quantization.keras import quantizers
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_configs
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_transforms
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import model_transformer
from tensorflow_model_optimization.python.core.quantization.keras.layers import conv_batchnorm_test_utils

ModelTransformer = model_transformer.ModelTransformer

Conv2DModel = conv_batchnorm_test_utils.Conv2DModel
DepthwiseConv2DModel = conv_batchnorm_test_utils.DepthwiseConv2DModel

keras = tf.keras

Conv2DBatchNormActivationQuantize = default_8bit_transforms.Conv2DBatchNormActivationQuantize
Conv2DBatchNormReLUQuantize = default_8bit_transforms.Conv2DBatchNormReLUQuantize


# TODO(alanchiao): reduce redundancy by parameterizing on Depthwise vs Conv.
class DefaultTransformsTest(tf.test.TestCase, parameterized.TestCase):

  def testTransformsConvBNReLUPattern(self):
    model = Conv2DModel.get_nonfolded_batchnorm_model(
        post_bn_activation=keras.layers.ReLU(6.0), model_type='functional')
    folded_model = Conv2DModel.get_folded_batchnorm_model(
        post_bn_activation=keras.layers.ReLU(6.0), is_quantized=True)

    transformed_model, _ = ModelTransformer(
        model,
        [default_8bit_transforms.Conv2DBatchNormReLU6Fold()]).transform()

    inputs = np.random.standard_normal(Conv2DModel.get_batched_input_shape())
    self.assertAllClose(
        transformed_model.predict(inputs), folded_model.predict(inputs))

  def testTransformsConvBNReLUPatternPreservesWeights(self):
    # random_init to prevent non-random initialization in resulting
    # in same weights between transformed and non-transformed models.
    model = Conv2DModel.get_nonfolded_batchnorm_model(
        post_bn_activation=keras.layers.ReLU(6.0),
        model_type='functional',
        random_init=True)

    transformed_model, _ = ModelTransformer(
        model,
        [default_8bit_transforms.Conv2DBatchNormReLU6Fold()]).transform()

    transformed_weights = transformed_model.get_weights()
    # Remove quantization related weights.
    del transformed_weights[3:8]

    self.assertEqual(len(transformed_weights), len(model.get_weights()))
    for i in range(len(transformed_weights)):
      self.assertAllEqual(transformed_weights[i], model.get_weights()[i])

  def testTransformsConvBNPattern(self):
    model = Conv2DModel.get_nonfolded_batchnorm_model(
        model_type='functional')
    folded_model = Conv2DModel.get_folded_batchnorm_model(
        is_quantized=True)

    with quantize.quantize_scope():
      transformed_model, _ = ModelTransformer(
          model, [default_8bit_transforms.Conv2DBatchNormFold()]).transform()

    inputs = np.random.standard_normal(Conv2DModel.get_batched_input_shape())
    self.assertAllClose(
        transformed_model.predict(inputs), folded_model.predict(inputs))

  def testTransformsConvBNPatternPreservesWeights(self):
    # random_init to prevent non-random initialization in resulting
    # in same weights between transformed and non-transformed models.
    model = Conv2DModel.get_nonfolded_batchnorm_model(
        model_type='functional',
        random_init=True)

    transformed_model, _ = ModelTransformer(
        model, [default_8bit_transforms.Conv2DBatchNormFold()]).transform()

    transformed_weights = transformed_model.get_weights()
    # Remove quantization related weights.
    del transformed_weights[3:8]

    self.assertEqual(len(transformed_weights), len(model.get_weights()))
    for i in range(len(transformed_weights)):
      self.assertAllEqual(transformed_weights[i], model.get_weights()[i])

  def testTransformsDepthwiseConvBNReLUPattern(self):
    model = DepthwiseConv2DModel.get_nonfolded_batchnorm_model(
        post_bn_activation=keras.layers.ReLU(6.0), model_type='functional')
    folded_model = DepthwiseConv2DModel.get_folded_batchnorm_model(
        post_bn_activation=keras.layers.ReLU(6.0), is_quantized=True)

    transformed_model, _ = ModelTransformer(
        model, [default_8bit_transforms.DepthwiseConv2DBatchNormReLU6Fold()
               ]).transform()

    inputs = np.random.standard_normal(
        DepthwiseConv2DModel.get_batched_input_shape())
    self.assertAllClose(
        transformed_model.predict(inputs), folded_model.predict(inputs))

  def testTransformsDepthwiseConvBNReLUPatternPreservesWeights(self):
    # random_init to prevent non-random initialization in resulting
    # in same weights between transformed and non-transformed models.
    model = DepthwiseConv2DModel.get_nonfolded_batchnorm_model(
        post_bn_activation=keras.layers.ReLU(6.0),
        model_type='functional',
        random_init=True)

    transformed_model, _ = ModelTransformer(
        model, [default_8bit_transforms.DepthwiseConv2DBatchNormReLU6Fold()
               ]).transform()

    transformed_weights = transformed_model.get_weights()
    # Remove quantization related weights.
    del transformed_weights[3:8]

    self.assertEqual(len(transformed_weights), len(model.get_weights()))
    for i in range(len(transformed_weights)):
      self.assertAllEqual(transformed_weights[i], model.get_weights()[i])

  @staticmethod
  def _get_model(layer_type, activation_type):
    activation = None
    if activation_type == 'relu':
      activation = keras.layers.ReLU(6.0)
    elif activation_type == 'act_relu':
      activation = keras.layers.Activation('relu')

    if layer_type == 'Conv2D':
      return Conv2DModel.get_nonfolded_batchnorm_model(
          model_type='functional', post_bn_activation=activation)
    elif layer_type == 'DepthwiseConv2D':
      return DepthwiseConv2DModel.get_nonfolded_batchnorm_model(
          model_type='functional', post_bn_activation=activation)

  @staticmethod
  def _get_input_shape(layer_type):
    if layer_type == 'Conv2D':
      return Conv2DModel.get_batched_input_shape()
    elif layer_type == 'DepthwiseConv2D':
      return DepthwiseConv2DModel.get_batched_input_shape()

  @parameterized.parameters('Conv2D', 'DepthwiseConv2D')
  def testConv2DBatchNormQuantize(self, layer_type):
    model = self._get_model(layer_type, False)
    input_shape = self._get_input_shape(layer_type)

    transformed_model, updated_metadata = ModelTransformer(
        model,
        [default_8bit_transforms.Conv2DBatchNormQuantize()],
    ).transform()

    conv_layer = transformed_model.layers[1]
    bn_layer = transformed_model.layers[2]

    self.assertIsInstance(
        conv_layer.activation, quantize_aware_activation.NoOpActivation)
    self.assertIsInstance(
        updated_metadata.get(bn_layer.name).get('quantize_config'),
        default_8bit_quantize_configs.Default8BitOutputQuantizeConfig)

    inputs = np.random.standard_normal(input_shape)
    self.assertAllClose(
        transformed_model.predict(inputs), model.predict(inputs))

  @parameterized.parameters(
      ('Conv2D', 'relu', Conv2DBatchNormReLUQuantize),
      ('Conv2D', 'act_relu', Conv2DBatchNormActivationQuantize),
      ('DepthwiseConv2D', 'relu', Conv2DBatchNormReLUQuantize),
      ('DepthwiseConv2D', 'act_relu',
       Conv2DBatchNormActivationQuantize),
  )
  def testConv2DBatchNormReLUQuantize(
      self, layer_type, activation_type, transform_type):
    model = self._get_model(layer_type, activation_type)
    input_shape = self._get_input_shape(layer_type)

    transformed_model, updated_metadata = ModelTransformer(
        model,
        [transform_type()],
    ).transform()

    conv_layer = transformed_model.layers[1]
    bn_layer = transformed_model.layers[2]

    self.assertIsInstance(
        conv_layer.activation, quantize_aware_activation.NoOpActivation)
    self.assertIsInstance(
        updated_metadata.get(bn_layer.name).get('quantize_config'),
        default_8bit_quantize_configs.NoOpQuantizeConfig)

    inputs = np.random.standard_normal(input_shape)
    self.assertAllClose(
        transformed_model.predict(inputs), model.predict(inputs))

  @parameterized.named_parameters(
      ('padding_valid', {'padding': 'valid'}),
      ('padding_same', {'padding': 'same'}),
      ('padding_same_dilation_2', {'padding': 'same', 'dilation_rate': 2}),
      ('strides', {'strides': 2}),
      ('dilation_rate', {'dilation_rate': 2}),
      ('depth_multiplier', {'depth_multiplier': 2}),
      ('regularizer', {
          'depthwise_regularizer': 'l2',
          'pointwise_regularizer': 'l2',
          'bias_regularizer': 'l2',
          'activity_regularizer': 'l2'}),
      ('constraint', {
          'depthwise_constraint': tf.keras.constraints.max_norm(2.),
          'pointwise_constraint': tf.keras.constraints.min_max_norm(0., 2.),
          'bias_constraint': tf.keras.constraints.unit_norm()})
  )
  def testSeparableConvQuantize_(self, kwargs):
    kwargs['filters'] = 2
    kwargs['kernel_size'] = 3
    num_samples = 2
    stack_size = 3
    num_row = 7
    num_col = 6

    sepconv_model = tf.keras.Sequential([
        tf.keras.Input(
            shape=(num_row, num_col, stack_size), batch_size=num_samples),
        tf.keras.layers.SeparableConv2D(**kwargs)])

    transformed_model, updated_metadata = ModelTransformer(
        sepconv_model,
        [default_8bit_transforms.SeparableConvQuantize()],
    ).transform()

    self.assertContainsSubset(
        updated_metadata.keys(), {'depthwise_conv2d', 'conv2d'})
    # Transformed model should have the same output shape
    self.assertEqual(sepconv_model.output_shape, transformed_model.output_shape)

    x = np.random.rand(*sepconv_model.input_shape)
    y = np.random.rand(*sepconv_model.output_shape)

    # Ensure model is equivalent, and forward pass results are the same.
    self.assertAllClose(sepconv_model.predict(x), transformed_model.predict(x))

    # Ensure model is equivalent, and training results are the same.
    sepconv_model.compile(loss='categorical_crossentropy', optimizer='sgd')
    sepconv_model.fit(x, y, epochs=100)
    transformed_model.compile(loss='categorical_crossentropy', optimizer='sgd')
    transformed_model.fit(x, y, epochs=100)

    # Over a long training cycle with constraints and regularizers, the model
    # can build very minute differences. Hence reducing tol to 1e-5.
    self.assertAllClose(sepconv_model.predict(x), transformed_model.predict(x),
                        atol=1e-5, rtol=1e-5)

  @parameterized.parameters(
      ('relu', default_8bit_transforms.AddReLUQuantize),
      ('act_relu', default_8bit_transforms.AddActivationQuantize),
  )
  def testAddReLUQuantize(self, activation_type, transform_type):
    add = keras.layers.Add()
    if activation_type == 'relu':
      activation = keras.layers.ReLU(6.0)
    elif activation_type == 'act_relu':
      activation = keras.layers.Activation('relu')

    inp1 = keras.layers.Input((3,))
    inp2 = keras.layers.Input((3,))
    x = activation(add([inp1, inp2]))
    model = keras.Model([inp1, inp2], x)

    transformed_model, updated_metadata = ModelTransformer(
        model,
        [transform_type()],
    ).transform()

    add_layer = transformed_model.layers[2]

    self.assertIsInstance(
        updated_metadata.get(add_layer.name).get('quantize_config'),
        default_8bit_quantize_configs.NoOpQuantizeConfig)

  def testAddsQuantizeLayerAfterInputLayer(self):
    inp1 = keras.layers.Input((3,))
    inp2 = keras.layers.Input((3,))
    x = keras.layers.Concatenate()([inp1, inp2])
    model = keras.Model([inp1, inp2], x)

    transformed_model, _ = ModelTransformer(
        model,
        [default_8bit_transforms.InputLayerQuantize()]).transform()

    for input_layer in transformed_model._input_layers:
      layer_after_input = input_layer._outbound_nodes[0].outbound_layer
      self.assertIsInstance(
          layer_after_input,
          quantize_layer.QuantizeLayer)
      self.assertIsInstance(
          layer_after_input.quantizer, quantizers.AllValuesQuantizer)

  def testConcatTransform(self):
    r"""Tests the Concat Transform.

               Input
              /     \
         Dense       Dense
             \      /
              Concat

      One Dense layer has a pre-specified QuantizeConfig, whereas the other does
      not. The Transform should ensure both the output FakeQuants are disabled,
      and only a FakeQuant after Concat is present.
    """
    dense_1 = keras.layers.Dense(3)
    dense_2 = keras.layers.Dense(3)
    concat = keras.layers.Concatenate()

    inp = keras.layers.Input((2,))
    x1 = dense_1(inp)
    x2 = dense_2(inp)
    x = concat([x1, x2])
    model = keras.Model(inp, x)

    layer_metadata = {
        # dense_1 has an existing quantize_config.
        dense_1.name: {
            'quantize_config':
                default_8bit_quantize_configs.Default8BitOutputQuantizeConfig()
        }
    }
    _, updated_metadata = ModelTransformer(
        model, [default_8bit_transforms.ConcatTransform()],
        layer_metadata=layer_metadata).transform()

    concat_quantize_config = updated_metadata.get(
        concat.name).get('quantize_config')
    # Concat should quantize the output.
    self.assertIsInstance(
        concat_quantize_config,
        default_8bit_quantize_configs.Default8BitOutputQuantizeConfig)
    self.assertNotEmpty(concat_quantize_config.get_output_quantizers(None))

    dense_1_quantize_config = updated_metadata.get(
        dense_1.name).get('quantize_config')
    # The existing quantize_config should do nothing for outputs.
    self.assertIsInstance(
        dense_1_quantize_config,
        default_8bit_quantize_configs.Default8BitOutputQuantizeConfig)
    self.assertEmpty(dense_1_quantize_config.get_output_quantizers(None))

    dense_2_quantize_config = updated_metadata.get(
        dense_2.name).get('quantize_config')
    # The quantize_config from registry should do nothing at output.
    self.assertEqual('Default8BitQuantizeConfig',
                     dense_2_quantize_config.__class__.__name__)
    self.assertEmpty(dense_2_quantize_config.get_output_quantizers(None))

  def testConcatMultipleLevels(self):
    r"""Tests case when concats applied to concats.

            Input --------------.
           /      \      |      |
         Dense   Dense   |      |
            \    /       |      |
             Concat    Dense   Dense
                 \     /        |
                  Concat        |
                        \      /
                         Concat

    The last Concat layer should be quantized but the rest
    of the outputs should just feed into it.
    """
    inp = keras.layers.Input((3,))
    x1 = keras.layers.Dense(3)(inp)
    x2 = keras.layers.Dense(3)(inp)
    x3 = keras.layers.Dense(3)(inp)
    x4 = keras.layers.Dense(3)(inp)
    c1 = keras.layers.Concatenate()([x1, x2])
    c2 = keras.layers.Concatenate()([c1, x3])
    c3 = keras.layers.Concatenate()([c2, x4])
    model = keras.Model(inp, c3)
    model.summary()

    _, layer_metadata = ModelTransformer(
        model,
        [default_8bit_transforms.ConcatTransform()]).transform()

    for layer in model.layers[1:-1]:
      quantize_config = layer_metadata[layer.name].get('quantize_config')
      self.assertEmpty(quantize_config.get_output_quantizers(None))

    c3_layer = model.layers[-1]
    quantize_config = layer_metadata[c3_layer.name].get('quantize_config')
    self.assertIsInstance(
        quantize_config,
        default_8bit_quantize_configs.Default8BitOutputQuantizeConfig)
    self.assertNotEmpty(quantize_config.get_output_quantizers(None))


if __name__ == '__main__':
  tf.test.main()
