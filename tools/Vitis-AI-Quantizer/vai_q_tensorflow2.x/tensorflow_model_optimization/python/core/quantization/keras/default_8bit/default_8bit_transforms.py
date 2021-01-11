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
"""Default 8-bit transforms."""

import collections
import inspect

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras import quantize_layer
from tensorflow_model_optimization.python.core.quantization.keras import quantizers
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_configs
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import transforms
from tensorflow_model_optimization.python.core.quantization.keras.layers import conv_batchnorm

LayerNode = transforms.LayerNode
LayerPattern = transforms.LayerPattern

_ConvBatchNorm2D = conv_batchnorm._ConvBatchNorm2D  # pylint: disable=protected-access
_DepthwiseConvBatchNorm2D = conv_batchnorm._DepthwiseConvBatchNorm2D  # pylint: disable=protected-access

keras = tf.keras


def _get_conv_bn_layers(bn_layer_node):
  bn_layer = bn_layer_node.layer
  conv_layer = bn_layer_node.input_layers[0].layer
  return conv_layer, bn_layer


def _get_weights(bn_layer_node):
  """Returns weight values for fused layer, including copying original values in unfused version."""

  return collections.OrderedDict(
      list(bn_layer_node.input_layers[0].weights.items())
      + list(bn_layer_node.weights.items()))


def _get_params(conv_layer, bn_layer, relu_layer=None):
  """Retrieve conv_bn params within wrapped layers."""
  if 'use_bias' in conv_layer['config']:
    if conv_layer['config']['use_bias']:
      raise ValueError(
          'use_bias should not be set to True in a Conv layer when followed '
          'by BatchNormalization. The bias in the Conv would be redundant '
          'with the one in the BatchNormalization.')

    del conv_layer['config']['use_bias']

  if 'name' in bn_layer['config']:
    del bn_layer['config']['name']

  # TODO(pulkitb): remove key conflicts
  params = dict(
      list(conv_layer['config'].items()) + list(bn_layer['config'].items()))

  if relu_layer is not None:
    params['post_activation'] = keras.layers.deserialize(relu_layer)

  return params


def _get_layer_node(fused_layer, weights):
  layer_config = keras.layers.serialize(fused_layer)
  layer_config['name'] = layer_config['config']['name']
  # This config tracks which layers get quantized, and whether they have a
  # custom QuantizeConfig.
  layer_metadata = {'quantize_config': None}

  return LayerNode(layer_config, weights, metadata=layer_metadata)


class Conv2DBatchNormFold(transforms.Transform):
  """Conv2DBatchNormFold."""

  def pattern(self):
    return LayerPattern('BatchNormalization', {},
                        [LayerPattern('Conv2D', {}, [])])

  def replacement(self, match_layer):
    conv_layer, bn_layer = _get_conv_bn_layers(match_layer)

    fused_params = _get_params(conv_layer, bn_layer)
    fused_layer = _ConvBatchNorm2D(**fused_params)

    weights = _get_weights(match_layer)
    return _get_layer_node(fused_layer, weights)

  def custom_objects(self):
    return {'_ConvBatchNorm2D': _ConvBatchNorm2D}


class Conv2DBatchNormReLU6Fold(Conv2DBatchNormFold):
  """Conv2DBatchNormReLU6Fold."""

  def pattern(self):
    return LayerPattern('ReLU', {'max_value': 6}, [
        LayerPattern('BatchNormalization', {},
                     [LayerPattern('Conv2D', {}, [])])
    ])

  def replacement(self, match_layer):
    relu_layer = match_layer.layer
    conv_layer, bn_layer = _get_conv_bn_layers(match_layer.input_layers[0])

    fused_params = _get_params(conv_layer, bn_layer, relu_layer)
    fused_layer = _ConvBatchNorm2D(**fused_params)

    weights = _get_weights(match_layer.input_layers[0])
    return _get_layer_node(fused_layer, weights)


class DepthwiseConv2DBatchNormReLU6Fold(transforms.Transform):
  """DepthwiseConv2DBatchNormReLU6Fold."""

  def pattern(self):
    return LayerPattern('ReLU', {'max_value': 6}, [
        LayerPattern('BatchNormalization', {},
                     [LayerPattern('DepthwiseConv2D', {}, [])])
    ])

  def replacement(self, match_layer):
    relu_layer = match_layer.layer
    conv_layer, bn_layer = _get_conv_bn_layers(match_layer.input_layers[0])

    fused_params = _get_params(conv_layer, bn_layer, relu_layer)
    fused_layer = _DepthwiseConvBatchNorm2D(**fused_params)

    weights = _get_weights(match_layer.input_layers[0])
    return _get_layer_node(fused_layer, weights)

  def custom_objects(self):
    return {'_DepthwiseConvBatchNorm2D': _DepthwiseConvBatchNorm2D}


class Conv2DBatchNormQuantize(transforms.Transform):
  """Ensure FQ does not get placed between Conv and BatchNorm."""

  def pattern(self):
    return LayerPattern(
        'BatchNormalization',
        inputs=[LayerPattern(
            'Conv2D|DepthwiseConv2D', config={'activation': 'linear'})])

  @staticmethod
  def _get_quantize_config(layer_node):
    return layer_node.metadata.get('quantize_config')

  def _has_custom_quantize_config(self, *layer_nodes):
    for layer_node in layer_nodes:
      if self._get_quantize_config(layer_node) is not None:
        return True
    return False

  def replacement(self, match_layer):
    bn_layer_node, conv_layer_node = match_layer, match_layer.input_layers[0]

    if self._has_custom_quantize_config(bn_layer_node, conv_layer_node):
      return match_layer

    conv_layer_node.layer['config']['activation'] = \
      keras.activations.serialize(quantize_aware_activation.NoOpActivation())
    bn_layer_node.metadata['quantize_config'] = \
      default_8bit_quantize_configs.Default8BitOutputQuantizeConfig()

    return match_layer

  def custom_objects(self):
    return {
        'NoOpQuantizeConfig': default_8bit_quantize_configs.NoOpQuantizeConfig,
        'NoOpActivation': quantize_aware_activation.NoOpActivation
    }


class Conv2DBatchNormReLUQuantize(Conv2DBatchNormQuantize):
  """Ensure FQ does not get placed between Conv, BatchNorm and ReLU."""

  def pattern(self):
    return LayerPattern(
        # TODO(pulkitb): Enhance match to only occur for relu, relu1 and relu6
        'ReLU',
        inputs=[super(Conv2DBatchNormReLUQuantize, self).pattern()])

  def replacement(self, match_layer):
    relu_layer_node = match_layer
    bn_layer_node = relu_layer_node.input_layers[0]
    conv_layer_node = bn_layer_node.input_layers[0]

    if self._has_custom_quantize_config(relu_layer_node, bn_layer_node,
                                        conv_layer_node):
      return match_layer

    conv_layer_node.layer['config']['activation'] = \
      keras.activations.serialize(quantize_aware_activation.NoOpActivation())
    bn_layer_node.metadata['quantize_config'] = \
      default_8bit_quantize_configs.NoOpQuantizeConfig()

    return match_layer

  def custom_objects(self):
    return {
        'NoOpQuantizeConfig': default_8bit_quantize_configs.NoOpQuantizeConfig,
        'NoOpActivation': quantize_aware_activation.NoOpActivation
    }


class Conv2DBatchNormActivationQuantize(Conv2DBatchNormReLUQuantize):
  """Ensure FQ does not get placed between Conv, BatchNorm and ReLU."""

  def pattern(self):
    return LayerPattern(
        'Activation',
        config={'activation': 'relu'},
        inputs=[Conv2DBatchNormQuantize.pattern(self)])


class SeparableConvQuantize(transforms.Transform):
  """Break SeparableConv into a DepthwiseConv and Conv layer.

  SeparableConv is a composition of a DepthwiseConv and a Conv layer. For the
  purpose of quantization, a FQ operation needs to be placed between the output
  of DepthwiseConv and the following Conv.

  This is needed since there is a dynamic tensor in between the two layers, and
  it's range information needs to be captured by the FakeQuant op to ensure
  full int8 quantization of the layers is possible.

  Splitting the layer into 2 ensures that each individual layer is handled
  correctly with respect to quantization.
  """

  def pattern(self):
    return LayerPattern('SeparableConv2D')

  @staticmethod
  def _get_quantize_config(layer_node):
    return layer_node.metadata.get('quantize_config')

  def _has_custom_quantize_config(self, *layer_nodes):
    for layer_node in layer_nodes:
      if self._get_quantize_config(layer_node) is not None:
        return True
    return False

  def replacement(self, match_layer):
    if self._has_custom_quantize_config(match_layer):
      return match_layer

    sepconv_layer = match_layer.layer
    sepconv_weights = list(match_layer.weights.values())

    # TODO(pulkitb): SeparableConv has kwargs other than constructor args which
    # need to be handled.
    # Applicable to both layers: trainable, dtype, name
    # Applicable to dconv: input_dim, input_shape, batch_input_shape, batch_size
    # Needs special handling: weights
    # Unknown: dynamic, autocast

    dconv_layer = tf.keras.layers.DepthwiseConv2D(
        kernel_size=sepconv_layer['config']['kernel_size'],
        strides=sepconv_layer['config']['strides'],
        padding=sepconv_layer['config']['padding'],
        depth_multiplier=sepconv_layer['config']['depth_multiplier'],
        data_format=sepconv_layer['config']['data_format'],
        dilation_rate=sepconv_layer['config']['dilation_rate'],
        activation=None,
        use_bias=False,
        depthwise_initializer=sepconv_layer['config']['depthwise_initializer'],
        depthwise_regularizer=sepconv_layer['config']['depthwise_regularizer'],
        depthwise_constraint=sepconv_layer['config']['depthwise_constraint'],
        trainable=sepconv_layer['config']['trainable']
    )
    dconv_weights = collections.OrderedDict()
    dconv_weights['depthwise_kernel:0'] = sepconv_weights[0]
    dconv_layer_config = keras.layers.serialize(dconv_layer)
    dconv_layer_config['name'] = dconv_layer.name
    # Needed to ensure these new layers are considered for quantization.
    dconv_metadata = {'quantize_config': None}

    conv_layer = tf.keras.layers.Conv2D(
        filters=sepconv_layer['config']['filters'],
        kernel_size=(1, 1),  # (1,) * rank
        strides=(1, 1),
        padding='valid',
        data_format=sepconv_layer['config']['data_format'],
        dilation_rate=sepconv_layer['config']['dilation_rate'],
        groups=1,
        activation=sepconv_layer['config']['activation'],
        use_bias=sepconv_layer['config']['use_bias'],
        kernel_initializer=sepconv_layer['config']['pointwise_initializer'],
        bias_initializer=sepconv_layer['config']['bias_initializer'],
        kernel_regularizer=sepconv_layer['config']['pointwise_regularizer'],
        bias_regularizer=sepconv_layer['config']['bias_regularizer'],
        activity_regularizer=sepconv_layer['config']['activity_regularizer'],
        kernel_constraint=sepconv_layer['config']['pointwise_constraint'],
        bias_constraint=sepconv_layer['config']['bias_constraint'],
        trainable=sepconv_layer['config']['trainable']
    )
    conv_weights = collections.OrderedDict()
    conv_weights['kernel:0'] = sepconv_weights[1]
    conv_weights['bias:0'] = sepconv_weights[2]
    conv_layer_config = keras.layers.serialize(conv_layer)
    conv_layer_config['name'] = conv_layer.name
    # Needed to ensure these new layers are considered for quantization.
    conv_metadata = {'quantize_config': None}

    dconv_layer_node = LayerNode(
        dconv_layer_config, weights=dconv_weights, metadata=dconv_metadata)
    return LayerNode(
        conv_layer_config,
        weights=conv_weights,
        input_layers=[dconv_layer_node],
        metadata=conv_metadata)


class AddReLUQuantize(transforms.Transform):
  """Ensure FQ does not get placed between Add and ReLU."""

  def pattern(self):
    return LayerPattern('ReLU', inputs=[LayerPattern('Add')])

  def replacement(self, match_layer):
    relu_layer_node = match_layer
    add_layer_node = relu_layer_node.input_layers[0]

    add_layer_node.metadata['quantize_config'] = \
      default_8bit_quantize_configs.NoOpQuantizeConfig()

    return match_layer

  def custom_objects(self):
    return {
        'NoOpQuantizeConfig': default_8bit_quantize_configs.NoOpQuantizeConfig,
    }


class AddActivationQuantize(AddReLUQuantize):
  """Ensure FQ does not get placed between Add and ReLU."""

  def pattern(self):
    return LayerPattern(
        'Activation',
        config={'activation': 'relu'},
        inputs=[LayerPattern('Add')])


class InputLayerQuantize(transforms.Transform):
  """Quantizes InputLayer, by adding QuantizeLayer after it.

  InputLayer => InputLayer -> QuantizeLayer
  """

  def pattern(self):
    return LayerPattern('InputLayer')

  def replacement(self, match_layer):
    quant_layer = quantize_layer.QuantizeLayer(
        quantizers.AllValuesQuantizer(
            num_bits=8, per_axis=False, symmetric=False, narrow_range=False))
    layer_config = keras.layers.serialize(quant_layer)
    layer_config['name'] = quant_layer.name

    quant_layer_node = LayerNode(
        layer_config,
        input_layers=[match_layer])

    return quant_layer_node

  def custom_objects(self):
    return {
        'QuantizeLayer': quantize_layer.QuantizeLayer,
        'MovingAverageQuantizer': quantizers.MovingAverageQuantizer,
        'AllValuesQuantizer': quantizers.AllValuesQuantizer
    }


class ConcatTransform(transforms.Transform):
  """Transform for Concatenate. Quantize only after concatenation."""

  # pylint:disable=protected-access

  def pattern(self):
    # TODO(pulkitb): Write a clean way to handle arbitrary length patterns.
    return LayerPattern(
        'Concatenate', inputs=[LayerPattern('.*'), LayerPattern('.*')])

  def _get_layer_type(self, layer_class_name):
    keras_layers = inspect.getmembers(tf.keras.layers, inspect.isclass)
    for layer_name, layer_type in keras_layers:
      if layer_name == layer_class_name:
        return layer_type
    return None

  def _disable_output_quantize(self, quantize_config):
    # TODO(pulkitb): Disabling quantize_config may also require handling
    # activation quantizers. Handle that properly.
    quantize_config.get_output_quantizers = lambda layer: []

  def replacement(self, match_layer):
    concat_layer_node = match_layer
    feeding_layer_nodes = match_layer.input_layers

    default_registry = default_8bit_quantize_registry.QuantizeRegistry(
    )

    feed_quantize_configs = []
    for feed_layer_node in feeding_layer_nodes:
      quantize_config = feed_layer_node.metadata.get('quantize_config')
      if not quantize_config:
        layer_class = self._get_layer_type(feed_layer_node.layer['class_name'])
        if layer_class is None:
          # Concat has an input layer we don't recognize. Return.
          return match_layer

        if layer_class == keras.layers.Concatenate:
          # Input layer to Concat is also Concat. Don't quantize it.
          feed_layer_node.metadata['quantize_config'] = \
            default_8bit_quantize_configs.NoOpQuantizeConfig()
          continue

        if not default_registry._is_supported_layer(layer_class):
          # Feeding layer is not supported by registry
          return match_layer

        quantize_config = default_registry._get_quantize_config(layer_class)
        feed_layer_node.metadata['quantize_config'] = quantize_config

      feed_quantize_configs.append(quantize_config)

    # TODO(pulkitb): this currently only disables output quantize config, but
    # cannot properly handle if the FQ was added to the activation. Hand this
    # properly.
    for quantize_config in feed_quantize_configs:
      self._disable_output_quantize(quantize_config)

    if not concat_layer_node.metadata.get('quantize_config'):
      concat_layer_node.metadata['quantize_config'] = \
        default_8bit_quantize_configs.Default8BitOutputQuantizeConfig()

    return concat_layer_node

  # pylint:enable=protected-access


class ConcatTransform3Inputs(ConcatTransform):

  def pattern(self):
    return LayerPattern(
        'Concatenate',
        inputs=[LayerPattern('.*'), LayerPattern('.*'), LayerPattern('.*')])


class ConcatTransform4Inputs(ConcatTransform):

  def pattern(self):
    return LayerPattern(
        'Concatenate',
        inputs=[LayerPattern('.*'), LayerPattern('.*'), LayerPattern('.*'),
                LayerPattern('.*')])


class ConcatTransform5Inputs(ConcatTransform):

  def pattern(self):
    return LayerPattern(
        'Concatenate',
        inputs=[LayerPattern('.*'), LayerPattern('.*'), LayerPattern('.*'),
                LayerPattern('.*'), LayerPattern('.*')])


class ConcatTransform6Inputs(ConcatTransform):

  def pattern(self):
    return LayerPattern(
        'Concatenate',
        inputs=[LayerPattern('.*'), LayerPattern('.*'), LayerPattern('.*'),
                LayerPattern('.*'), LayerPattern('.*'), LayerPattern('.*')])
