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
"""Quantization registry which specifies how layers should be quantized."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantize_config
from tensorflow_model_optimization.python.core.quantization.keras import quantize_registry
from tensorflow_model_optimization.python.core.quantization.keras import quantizers
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_configs
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantizers
from tensorflow_model_optimization.python.core.quantization.keras.layers import conv_batchnorm

QuantizeConfig = quantize_config.QuantizeConfig

layers = tf.keras.layers


class _QuantizeInfo(object):
  """QuantizeInfo."""

  def __init__(self,
               layer_type,
               weight_attrs,
               activation_attrs,
               quantize_output=False):
    """QuantizeInfo.

    Args:
      layer_type: Type of keras layer.
      weight_attrs: List of quantizable weight attributes of layer.
      activation_attrs: List of quantizable activation attributes of layer.
      quantize_output: Bool. Should we quantize the output of the layer.
    """
    self.layer_type = layer_type
    self.weight_attrs = weight_attrs
    self.activation_attrs = activation_attrs
    self.quantize_output = quantize_output


def _no_quantize(layer_type):
  return _QuantizeInfo(layer_type, [], [], False)


class _RNNHelper(object):
  """Helper functions for working with RNNs."""

  def _get_rnn_cells(self, rnn_layer):
    """Returns the list of cells in an RNN layer."""
    if isinstance(rnn_layer.cell, layers.StackedRNNCells):
      return rnn_layer.cell.cells
    else:
      return [rnn_layer.cell]


class QuantizeRegistry(quantize_registry.QuantizeRegistry, _RNNHelper):
  """QuantizationRegistry for built-in Keras classes for default 8-bit scheme."""

  # TODO(tfmot): expand layers test in quantize_functional_test.py
  # to add more layers to allowlist.
  _LAYER_QUANTIZE_INFO = [
      # Activation Layers
      _QuantizeInfo(layers.ReLU, [], [], True),
      _QuantizeInfo(layers.Softmax, [], []),
      # Enable once verified.
      # layers.ELU,
      # layers.LeakyReLU,
      # layers.PReLU,
      # layers.ThresholdedReLU,

      # Convolution Layers
      # _QuantizeInfo(layers.Conv1D, ['kernel'], ['activation']),

      # layers.Conv2D is supported and handled in code below.
      # layers.DepthwiseConv2D is supported and handled in code below.

      # _QuantizeInfo(layers.Conv3D, ['kernel'], ['activation']),
      # _QuantizeInfo(layers.Conv2DTranspose, ['kernel'], ['activation']),
      # _QuantizeInfo(layers.Conv3DTranspose, ['kernel'], ['activation']),
      _no_quantize(layers.Cropping1D),
      _no_quantize(layers.Cropping2D),
      _no_quantize(layers.Cropping3D),
      # _no_quantize(layers.UpSampling1D),
      # _no_quantize(layers.UpSampling2D),
      # _no_quantize(layers.UpSampling3D),
      _no_quantize(layers.ZeroPadding1D),
      _no_quantize(layers.ZeroPadding2D),
      # _no_quantize(layers.ZeroPadding3D),
      # Enable once verified.
      # layers.SeparableConv1D,
      # layers.SeparableConv2D,

      # Core Layers
      _no_quantize(layers.ActivityRegularization),
      _QuantizeInfo(layers.Dense, ['kernel'], ['activation']),
      _no_quantize(layers.Dropout),
      _no_quantize(layers.Flatten),
      # _no_quantize(layers.Masking),
      _no_quantize(layers.Permute),
      # _no_quantize(layers.RepeatVector),
      _no_quantize(layers.Reshape),
      _no_quantize(layers.SpatialDropout1D),
      _no_quantize(layers.SpatialDropout2D),
      _no_quantize(layers.SpatialDropout3D),
      # layers.Lambda needs custom handling by the user.

      # Pooling Layers
      _QuantizeInfo(layers.AveragePooling1D, [], [], True),
      _QuantizeInfo(layers.AveragePooling2D, [], [], True),
      # _QuantizeInfo(layers.AveragePooling3D, [], [], True),
      _QuantizeInfo(layers.GlobalAveragePooling1D, [], [], True),
      _QuantizeInfo(layers.GlobalAveragePooling2D, [], [], True),
      _QuantizeInfo(layers.GlobalAveragePooling3D, [], [], True),
      _no_quantize(layers.GlobalMaxPooling1D),
      _no_quantize(layers.GlobalMaxPooling2D),
      _no_quantize(layers.GlobalMaxPooling3D),
      # _no_quantize(layers.MaxPooling1D),
      _no_quantize(layers.MaxPooling2D),
      # _no_quantize(layers.MaxPooling3D),

      # _QuantizeInfo(layers.LocallyConnected1D, ['kernel'], ['activation']),
      # _QuantizeInfo(layers.LocallyConnected2D, ['kernel'], ['activation']),
      _QuantizeInfo(layers.Add, [], [], True),

      # Enable once verified with TFLite behavior.
      # layers.Embedding: ['embeddings'],

      # BatchNormalization is handled elsewhere, in the cases
      # where it's preceded by convolutional layers or a Dense layer.
      #   layers.BatchNormalization: [],

      # Merge layers to be added.

      # RNN Cells
      # TODO(pulkitb): Verify RNN layers behavior.
      # TODO(tfmot): check if we still need to allowlist via compat.v1 and
      # compat.v2 to support legacy TensorFlow 2.X
      # behavior where the v2 RNN uses the v1 RNNCell instead of the v2 RNNCell.
      # See b/145939875 for details.
      # _QuantizeInfo(tf.keras.layers.GRUCell, ['kernel', 'recurrent_kernel'],
      #               ['activation', 'recurrent_activation']),
      # _QuantizeInfo(tf.keras.layers.LSTMCell, ['kernel', 'recurrent_kernel'],
      #               ['activation', 'recurrent_activation']),
      # _QuantizeInfo(tf.keras.experimental.PeepholeLSTMCell,
      #               ['kernel', 'recurrent_kernel'],
      #               ['activation', 'recurrent_activation']),
      # _QuantizeInfo(tf.keras.layers.SimpleRNNCell,
      #               ['kernel', 'recurrent_kernel'],
      #               ['activation', 'recurrent_activation']),

      # TODO(tf-mot): Move layers out once Transforms indicate quantization.
      _no_quantize(conv_batchnorm._ConvBatchNorm2D),  # pylint: disable=protected-access
      _no_quantize(conv_batchnorm._DepthwiseConvBatchNorm2D),  # pylint: disable=protected-access
  ]

  def __init__(self):
    self._layer_quantize_map = {}
    for quantize_info in self._LAYER_QUANTIZE_INFO:
      self._layer_quantize_map[quantize_info.layer_type] = quantize_info

    # Hack for `Activation` layer. That is the only layer with a separate
    # QuantizeConfig.
    self._layer_quantize_map[
        layers.Activation] = Default8BitActivationQuantizeConfig()
    self._layer_quantize_map[layers.Conv2D] = Default8BitConvQuantizeConfig(
        ['kernel'], ['activation'], False)
    self._layer_quantize_map[
        layers.DepthwiseConv2D] = Default8BitConvQuantizeConfig(
            ['depthwise_kernel'], ['activation'], False)

  def _is_supported_layer(self, layer_class):
    return layer_class in self._layer_quantize_map

  def _is_rnn_layer(self, layer):
    return layer.__class__ in {
        layers.GRU,
        layers.LSTM,
        layers.RNN,
        layers.SimpleRNN,
    }

  def _get_quantize_info(self, layer_class):
    return self._layer_quantize_map[layer_class]

  # Interface functions.

  def supports(self, layer):
    """Returns whether the registry supports this layer type.

    # TODO(pulkitb): Consider pushing this function up to the registry.

    Args:
      layer: The layer to check for support.

    Returns:
      True/False whether the layer type is supported.

    """
    if self._is_supported_layer(layer.__class__):
      return True

    if self._is_rnn_layer(layer):
      for rnn_cell in self._get_rnn_cells(layer):
        # All cells in the RNN layer should be supported.
        if not self._is_supported_layer(rnn_cell.__class__):
          return False
      return True

    return False

  def _get_quantize_config(self, layer_type):
    quantize_info = self._get_quantize_info(layer_type)

    # In case of `Activation`, there is no `_QuantizeInfo` object. It
    # directly stores a `QuantizeConfig`.
    if isinstance(quantize_info, QuantizeConfig):
      return quantize_info

    return Default8BitQuantizeConfig(quantize_info.weight_attrs,
                                     quantize_info.activation_attrs,
                                     quantize_info.quantize_output)

  def get_quantize_config(self, layer):
    """Returns the quantization config for the given layer.

    Args:
      layer: input layer to return quantize config for.

    Returns:
      Returns the QuantizeConfig for the given layer.
    """
    if not self.supports(layer):
      raise ValueError(
          '`get_quantize_config()` called on an unsupported layer {}. Check '
          'if layer is supported by calling `supports()`. Alternatively, you '
          'can use `QuantizeConfig` to specify a behavior for your layer.'
          .format(layer.__class__))

    if self._is_supported_layer(layer.__class__):
      return self._get_quantize_config(layer.__class__)

    if self._is_rnn_layer(layer):
      weight_attrs = []
      activation_attrs = []
      for rnn_cell in self._get_rnn_cells(layer):
        quantize_info = self._get_quantize_info(rnn_cell.__class__)
        weight_attrs.append(quantize_info.weight_attrs)
        activation_attrs.append(quantize_info.activation_attrs)

      # Result quantization for RNN isn't straight-forward like regular layers.
      # To implement during full RNN support.
      return Default8BitQuantizeConfigRNN(weight_attrs, activation_attrs, False)

    # Should never come here.
    raise ValueError('Invalid Layer type {}'.format(layer.__class__))


class Default8BitQuantizeConfig(QuantizeConfig):
  """QuantizeConfig for non recurrent Keras layers."""

  def __init__(self, weight_attrs, activation_attrs, quantize_output):
    self.weight_attrs = weight_attrs
    self.activation_attrs = activation_attrs
    self.quantize_output = quantize_output

    # TODO(pulkitb): For some layers such as Conv2D, per_axis should be True.
    # Add mapping for which layers support per_axis.
    self.weight_quantizer = quantizers.LastValueQuantizer(
        num_bits=8, per_axis=False, symmetric=True, narrow_range=True)
    self.activation_quantizer = quantizers.MovingAverageQuantizer(
        num_bits=8, per_axis=False, symmetric=False, narrow_range=False)

  def get_weights_and_quantizers(self, layer):
    return [(getattr(layer, weight_attr), self.weight_quantizer)
            for weight_attr in self.weight_attrs]

  def get_activations_and_quantizers(self, layer):
    return [(getattr(layer, activation_attr), self.activation_quantizer)
            for activation_attr in self.activation_attrs]

  def set_quantize_weights(self, layer, quantize_weights):
    if len(self.weight_attrs) != len(quantize_weights):
      raise ValueError(
          '`set_quantize_weights` called on layer {} with {} '
          'weight parameters, but layer expects {} values.'.format(
              layer.name, len(quantize_weights), len(self.weight_attrs)))

    for weight_attr, weight in zip(self.weight_attrs, quantize_weights):
      current_weight = getattr(layer, weight_attr)
      if current_weight.shape != weight.shape:
        raise ValueError('Existing layer weight shape {} is incompatible with'
                         'provided weight shape {}'.format(
                             current_weight.shape, weight.shape))

      setattr(layer, weight_attr, weight)

  def set_quantize_activations(self, layer, quantize_activations):
    if len(self.activation_attrs) != len(quantize_activations):
      raise ValueError(
          '`set_quantize_activations` called on layer {} with {} '
          'activation parameters, but layer expects {} values.'.format(
              layer.name, len(quantize_activations),
              len(self.activation_attrs)))

    for activation_attr, activation in \
        zip(self.activation_attrs, quantize_activations):
      setattr(layer, activation_attr, activation)

  def get_output_quantizers(self, layer):
    if self.quantize_output:
      return [self.activation_quantizer]
    return []

  @classmethod
  def from_config(cls, config):
    """Instantiates a `Default8BitQuantizeConfig` from its config.

    Args:
        config: Output of `get_config()`.

    Returns:
        A `Default8BitQuantizeConfig` instance.
    """
    return cls(**config)

  def get_config(self):
    # TODO(pulkitb): Add weight and activation quantizer to config.
    # Currently it's created internally, but ideally the quantizers should be
    # part of the constructor and passed in from the registry.
    return {
        'weight_attrs': self.weight_attrs,
        'activation_attrs': self.activation_attrs,
        'quantize_output': self.quantize_output
    }

  def __eq__(self, other):
    if not isinstance(other, Default8BitQuantizeConfig):
      return False

    return (self.weight_attrs == other.weight_attrs and
            self.activation_attrs == self.activation_attrs and
            self.weight_quantizer == other.weight_quantizer and
            self.activation_quantizer == other.activation_quantizer and
            self.quantize_output == other.quantize_output)

  def __ne__(self, other):
    return not self.__eq__(other)


class Default8BitQuantizeConfigRNN(Default8BitQuantizeConfig, _RNNHelper):
  """QuantizeConfig for RNN layers."""

  def get_weights_and_quantizers(self, layer):
    weights_quantizers = []
    for weight_attrs_cell, rnn_cell in \
        zip(self.weight_attrs, self._get_rnn_cells(layer)):
      for weight_attr in weight_attrs_cell:
        weights_quantizers.append(
            (getattr(rnn_cell, weight_attr), self.weight_quantizer))

    return weights_quantizers

  def get_activations_and_quantizers(self, layer):
    activations_quantizers = []
    for activation_attrs_cell, rnn_cell in \
        zip(self.activation_attrs, self._get_rnn_cells(layer)):
      for activation_attr in activation_attrs_cell:
        activations_quantizers.append(
            (getattr(rnn_cell, activation_attr), self.activation_quantizer))

    return activations_quantizers

  def _flatten(self, list_of_lists):
    flat_list = []
    for sublist in list_of_lists:
      for item in sublist:
        flat_list.append(item)
    return flat_list

  def set_quantize_weights(self, layer, quantize_weights):
    flattened_weight_attrs = self._flatten(self.weight_attrs)
    if len(flattened_weight_attrs) != len(quantize_weights):
      raise ValueError(
          '`set_quantize_weights` called on layer {} with {} '
          'weight parameters, but layer expects {} values.'.format(
              layer.name, len(quantize_weights), len(flattened_weight_attrs)))

    i = 0
    for weight_attrs_cell, rnn_cell in \
        zip(self.weight_attrs, self._get_rnn_cells(layer)):
      for weight_attr in weight_attrs_cell:
        current_weight = getattr(rnn_cell, weight_attr)
        quantize_weight = quantize_weights[i]

        if current_weight.shape != quantize_weight.shape:
          raise ValueError('Existing layer weight shape {} is incompatible with'
                           'provided weight shape {}'.format(
                               current_weight.shape, quantize_weight.shape))

        setattr(rnn_cell, weight_attr, quantize_weight)
        i += 1

  def set_quantize_activations(self, layer, quantize_activations):
    flattened_activation_attrs = self._flatten(self.activation_attrs)
    if len(flattened_activation_attrs) != len(quantize_activations):
      raise ValueError(
          '`set_quantize_activations` called on layer {} with {} '
          'activation parameters, but layer expects {} values.'.format(
              layer.name, len(quantize_activations),
              len(flattened_activation_attrs)))

    i = 0
    for activation_attrs_cell, rnn_cell in \
        zip(self.activation_attrs, self._get_rnn_cells(layer)):
      for activation_attr in activation_attrs_cell:
        setattr(rnn_cell, activation_attr, quantize_activations[i])
        i += 1


class Default8BitActivationQuantizeConfig(QuantizeConfig):
  """QuantizeConfig for keras.layers.Activation.

  `keras.layers.Activation` needs a separate `QuantizeConfig` since the
  decision to quantize depends on the specific activation type.
  """

  def _assert_activation_layer(self, layer):
    if not isinstance(layer, layers.Activation):
      raise RuntimeError(
          'Default8BitActivationQuantizeConfig can only be used with '
          '`keras.layers.Activation`.')

  def get_weights_and_quantizers(self, layer):
    self._assert_activation_layer(layer)
    return []

  def get_activations_and_quantizers(self, layer):
    self._assert_activation_layer(layer)
    return []

  def set_quantize_weights(self, layer, quantize_weights):
    self._assert_activation_layer(layer)

  def set_quantize_activations(self, layer, quantize_activations):
    self._assert_activation_layer(layer)

  def get_output_quantizers(self, layer):
    self._assert_activation_layer(layer)

    if not hasattr(layer.activation, '__name__'):
      raise ValueError('Activation {} not supported by '
                       'Default8BitActivationQuantizeConfig.'.format(
                           layer.activation))

    if layer.activation.__name__ in ['relu']:
      # 'relu' should generally get fused into the previous layer.
      return [quantizers.MovingAverageQuantizer(
          num_bits=8, per_axis=False, symmetric=False, narrow_range=False)]
    elif layer.activation.__name__ in ['linear', 'softmax']:
      return []

    raise ValueError('Activation {} not supported by '
                     'Default8BitActivationQuantizeConfig.'.format(
                         layer.activation))

  def get_config(self):
    return {}


class Default8BitConvQuantizeConfig(Default8BitQuantizeConfig):
  """QuantizeConfig for Conv2D/DepthwiseConv2D layers."""

  def __init__(self, weight_attrs, activation_attrs, quantize_output):
    super(Default8BitConvQuantizeConfig,
          self).__init__(weight_attrs, activation_attrs, quantize_output)

    self.weight_quantizer = default_8bit_quantizers.Default8BitConvWeightsQuantizer(
    )


def _types_dict():
  return {
      'Default8BitQuantizeConfig':
          Default8BitQuantizeConfig,
      'Default8BitQuantizeConfigRNN':
          Default8BitQuantizeConfigRNN,
      'Default8BitActivationQuantizeConfig':
          Default8BitActivationQuantizeConfig,
      'Default8BitConvQuantizeConfig':
          Default8BitConvQuantizeConfig,
      'NoOpQuantizeConfig':
          default_8bit_quantize_configs.NoOpQuantizeConfig,
      'Default8BitOutputQuantizeConfig':
          default_8bit_quantize_configs.Default8BitOutputQuantizeConfig
  }
