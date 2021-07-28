# Copyright 2019 Xilinx Inc.
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
"""Vitis Conv-Batchnorm quantize layers."""

import tensorflow as tf

from tensorflow import keras
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import register_keras_serializable

from tensorflow_model_optimization.python.core.quantization.keras.vitis.base import quantizer as quantizer_mod
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

__all__ = [
    'VitisConvBN', 'VitisConvBNQuantize', 'VitisDepthwiseConvBN',
    'VitisDepthwiseConvBNQuantize'
]

serialize_keras_object = tf.keras.utils.serialize_keras_object
deserialize_keras_object = tf.keras.utils.deserialize_keras_object
logger = common_utils.VAILogger


def _get_folded_kernel_bias(conv_type, kernel, bias, mu, var, gamma, beta,
                            epsilon):
  """ Get folded kernel and bias
      folded_kernel = kernel * multiplier
                    = kernel * gamma / sigma_bt

      folded_bias = beta - (mu - bias) * multiplier
                  = beta - (mu - bias) * gamma / sigma
  """
  sigma = math_ops.rsqrt(var + epsilon)
  if gamma is not None:
    multiplier = math_ops.mul(gamma, sigma)
  else:
    multiplier = sigma
  if conv_type == 'DepthwiseConv2D':
    new_shape = [kernel.shape[2], kernel.shape[3]]
    depthwise_multiplier = array_ops.reshape(multiplier, new_shape)
    folded_kernel = math_ops.mul(
        depthwise_multiplier, kernel, name='depthwise_kernel')
  else:
    folded_kernel = math_ops.mul(multiplier, kernel, name='kernel')

  folded_bias = math_ops.subtract(beta, (mu - bias) * multiplier, name='bias')
  return folded_kernel, folded_bias


def _get_bn_correction(conv_type, kernel, bias, mu_bt, var_bt, mu_mv, var_mv,
                       gamma, epsilon):
  """Get batchnorm correction params.

     Before freeze:
       corr_scale = sigma_bt / sigma_mv
       corr_recip = 1 / corr_scale
       corr_offset = 0
     After freeze:
       corr_scale = sigma_bt / sigma_mv
       corr_recip = 1
       corr_offset = gamma * ( (mu_bt - bias)/sigma_bt - (mu_mv - bias)/sigma_mv)
  """
  sigma_bt = math_ops.rsqrt(var_bt + epsilon)
  sigma_bt_recip = math_ops.reciprocal(sigma_bt)
  sigma_mv = math_ops.rsqrt(var_mv + epsilon)
  sigma_mv_recip = math_ops.reciprocal(sigma_mv)

  corr_scale = math_ops.divide(sigma_bt, sigma_mv, name='corr_scale')
  corr_recip = math_ops.reciprocal(corr_scale)
  corr_offset = array_ops.zeros(mu_bt.shape)

  if conv_type == 'DepthwiseConv2D':
    new_shape = [kernel.shape[2], kernel.shape[3]]
    corr_scale = array_ops.reshape(corr_scale, new_shape)

  return corr_scale, corr_recip, corr_offset


@register_keras_serializable(package='Vitis', name='VitisConvBN')
class VitisConvBN(tf.keras.layers.Layer):
  """A Wrapper layer of folded convolution and batchnorm."""

  def __init__(self, conv_layer, bn_layer, activation=None, **kwargs):
    assert isinstance(conv_layer, tf.keras.layers.Conv2D)
    assert isinstance(bn_layer, tf.keras.layers.BatchNormalization)
    #(TODO) check conv activation.
    self.conv_layer = conv_layer
    self.bn_layer = bn_layer
    self.activation = activations.get(activation)

    if 'name' not in kwargs:
      kwargs['name'] = self._make_layer_name(conv_layer, bn_layer)

    super(VitisConvBN, self).__init__(**kwargs)

    self._track_trackable(conv_layer, name='conv_layer')
    self._track_trackable(bn_layer, name='bn_layer')

  @staticmethod
  def _make_layer_name(conv_layer, bn_layer):
    layer_name = '{}_{}_{}'.format('quant', conv_layer.name, bn_layer.name)
    return layer_name

  def build(self, input_shape=None):
    super(VitisConvBN, self).build(input_shape)
    if not self.conv_layer.built or not self.bn_layer.built:
      self.conv_layer.build(input_shape)
      conv_out_shape = self.conv_layer.compute_output_shape(input_shape)
      self.bn_layer.build(conv_out_shape)
    self.built = True

  def call(self, inputs, training=None):
    if training is None:
      training = tf.keras.backend.learning_phase()

    outputs = self.conv_layer.call(inputs)
    outputs = self.bn_layer.call(outputs, training=training)
    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def get_config(self):
    base_config = super(VitisConvBN, self).get_config()
    config = {
        'conv_layer': serialize_keras_object(self.conv_layer),
        'bn_layer': serialize_keras_object(self.bn_layer),
        'activation': activations.serialize(self.activation),
    }
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    config = config.copy()
    conv_layer = tf.keras.layers.deserialize(config.pop('conv_layer'))
    bn_layer = tf.keras.layers.deserialize(config.pop('bn_layer'))
    activation = config.pop('activation')

    return cls(
        conv_layer=conv_layer,
        bn_layer=bn_layer,
        activation=activation,
        **config)


@register_keras_serializable(package='Vitis', name='VitisConvBNQuantize')
class VitisConvBNQuantize(tf.keras.layers.Layer):
  """A Wrapper layer emulate quantization of folded convolution and batchnorm."""

  def __init__(self, conv_layer, bn_layer, activation, freeze_bn_delay,
               quantize_config, mode, **kwargs):
    assert isinstance(conv_layer, tf.keras.layers.Conv2D)
    assert isinstance(bn_layer, tf.keras.layers.BatchNormalization)
    #(TODO) check conv activation.
    self.conv_layer = conv_layer
    self.bn_layer = bn_layer
    self.activation = activations.get(activation)

    if 'name' not in kwargs:
      kwargs['name'] = self._make_layer_name(conv_layer, bn_layer)

    # Prepare the params for the folded convolution
    self.rank = self.conv_layer.rank
    self.filters = self.conv_layer.filters
    self._is_causal = self.conv_layer._is_causal
    self._compute_causal_padding = self.conv_layer._compute_causal_padding
    self._channels_first = self.conv_layer._channels_first
    self._tf_data_format = self.conv_layer._tf_data_format

    super(VitisConvBNQuantize, self).__init__(**kwargs)
    self.quantize_config = quantize_config
    self._mode = mode
    if freeze_bn_delay is not None:
      self.freeze_bn_delay = int(freeze_bn_delay)
    else:
      self.freeze_bn_delay = None

    self._track_trackable(conv_layer, name='conv_layer')
    self._track_trackable(bn_layer, name='bn_layer')

  @staticmethod
  def _make_layer_name(conv_layer, bn_layer):
    layer_name = '{}_{}_{}'.format('quant', conv_layer.name, bn_layer.name)
    return layer_name

  def _get_shape_map(self):
    return {
        'kernel': self.conv_layer.kernel.shape,
        'bias': self.bn_layer.beta.shape
    }

  def _build_for_quantization(self):
    """All Keras build() logic for quantization for fused layers."""
    self.optimizer_step = self.add_weight(
        'optimizer_step',
        initializer=keras.initializers.Constant(0),
        dtype=dtypes.int32,
        trainable=False)

    # Build weight quantizer variables with shape of conv layer kernel and bn layer gamma.
    shape_map = self._get_shape_map()
    quantizable_weights = self.quantize_config.get_quantizable_weights()
    weight_quantizers = self.quantize_config.get_weight_quantizers()

    self._weight_vars = []
    for weight, quantizer in zip(quantizable_weights, weight_quantizers):
      if weight not in shape_map:
        logger.error('Fail to get shape for {} of layer {}.'.format(
            weight, self))
      quantizer_vars = quantizer.build(shape_map[weight], weight, self)
      self._weight_vars.append((weight, quantizer, quantizer_vars))

    # Build activation quantizers
    self._quantize_activations = []
    for activation, quantizer in self.quantize_config.get_activations_and_quantizers(
        self):
      quantize_activation = vitis_quantize_aware_activation.QuantizeAwareActivation(
          activation, quantizer, self.mode, self.optimizer_step, self)
      self._quantize_activations.append(quantize_activation)

  def build(self, input_shape=None):
    super(VitisConvBNQuantize, self).build(input_shape)
    if not self.conv_layer.built or not self.bn_layer.built:
      self.conv_layer.build(input_shape)
      conv_out_shape = self.conv_layer.compute_output_shape(input_shape)
      self.bn_layer.build(conv_out_shape)

    self._convolution_op = self.conv_layer._convolution_op
    # The folded conv always have bias
    self.use_bias = True
    self.built = True

    self._build_for_quantization()

  def _make_quantizer_fn(self, quantizer, x, training, mode, quantizer_vars):
    """Use currying to return True/False specialized fns to the cond."""

    def quantizer_fn():
      return quantizer(x, training, mode, weights=quantizer_vars)

    return quantizer_fn

  def _get_batch_mean_var(self, inputs):
    # Get batch_mean and variance, here we use the codes copy from:
    # https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/python/keras/layers/normalization.py#L806
    input_shape = inputs.shape
    ndims = len(input_shape)
    reduction_axes = [i for i in range(ndims) if i not in self.bn_layer.axis]
    keep_dims = self.bn_layer.virtual_batch_size is not None or len(
        self.bn_layer.axis) > 1
    return self.bn_layer._moments(
        math_ops.cast(inputs, self.bn_layer._param_dtype),
        reduction_axes,
        keep_dims=keep_dims)

  def _quantize_weights(self, training):
    # Quantize the folded kernel and bias
    for weight, quantizer, quantizer_vars in self._weight_vars:
      weight_tensor = getattr(self, weight)
      quantized_weight = tf_utils.smart_cond(
          training,
          self._make_quantizer_fn(quantizer, weight_tensor, True, self.mode,
                                  quantizer_vars),
          self._make_quantizer_fn(quantizer, weight_tensor, False, self.mode,
                                  quantizer_vars))
      setattr(self, weight, quantized_weight)

  def _run_folded_conv(self, inputs, training):
    """Run folded convolution.

       The folded conv do not has real variables so it can not be created by
       normal keras.Conv2D layers. Here we use the codes copy from
       https://github.com/tensorflow/tensorflow/blob/9edbe5075f79a4a95ed14a2be831f9b59e61f49d/tensorflow/python/keras/layers/convolutional.py#L245
    """
    if self._is_causal:  # Apply causal padding to inputs for Conv1D.
      inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))

    outputs = self._convolution_op(inputs, self.kernel)
    return outputs

  def _run_folded_bias_add(self, inputs):
    outputs = inputs
    # Bias add
    if self.use_bias:
      output_rank = outputs.shape.rank
      if self.rank == 1 and self._channels_first:
        # nn.bias_add does not accept a 1D input tensor.
        bias = array_ops.reshape(self.bias, (1, self.filters, 1))
        outputs += bias
      else:
        # Handle multiple batch dimensions.
        if output_rank is not None and output_rank > 2 + self.rank:

          def _apply_fn(o):
            return nn.bias_add(o, self.bias, data_format=self._tf_data_format)

          outputs = nn_ops.squeeze_batch_dims(
              outputs, _apply_fn, inner_rank=self.rank + 1)
        else:
          outputs = nn.bias_add(
              outputs, self.bias, data_format=self._tf_data_format)
    return outputs

  def call(self, inputs, training=None):
    if training is None:
      training = tf.keras.backend.learning_phase()

    bias = tf_utils.smart_cond(self.conv_layer.use_bias,
                               lambda: self.conv_layer.bias, lambda: 0)

    if training:
      self.optimizer_step.assign_add(1)

    freeze_bn = tf_utils.smart_cond(
        self.freeze_bn_delay is not None, lambda: math_ops.greater_equal(
            self.optimizer_step, self.freeze_bn_delay), lambda: False)
    #  tf.print('step: {}, freeze_bn: {}'.format(self.optimizer_step, freeze_bn))

    if training and not freeze_bn:
      # Run float conv and bn to update the moving mean and variance
      conv_out = self.conv_layer.call(inputs)
      bn_out = self.bn_layer.call(conv_out, training=training)

      mu_bt, var_bt = self._get_batch_mean_var(conv_out)

      self.kernel, self.bias = _get_folded_kernel_bias(
          conv_type='Conv2D',
          kernel=self.conv_layer.kernel,
          bias=bias,
          mu=mu_bt,
          var=var_bt,
          gamma=self.bn_layer.gamma,
          beta=self.bn_layer.beta,
          epsilon=self.bn_layer.epsilon)

      # BatchNorm Correction
      corr_scale, corr_recip, corr_offset = _get_bn_correction(
          conv_type='Conv2D',
          kernel=self.conv_layer.kernel,
          bias=bias,
          mu_bt=mu_bt,
          var_bt=var_bt,
          mu_mv=self.bn_layer.moving_mean,
          var_mv=self.bn_layer.moving_variance,
          gamma=self.bn_layer.gamma,
          epsilon=self.bn_layer.epsilon)

      self.kernel = math_ops.mul(self.kernel, corr_scale)
      self.bias = math_ops.add(self.bias, corr_offset)

      self._quantize_weights(training)
      outputs = self._run_folded_conv(inputs, training)
      # BatchNorm Correction for convolution outputs
      outputs = math_ops.mul(outputs, corr_recip)
    else:
      self.kernel, self.bias = _get_folded_kernel_bias(
          conv_type='Conv2D',
          kernel=self.conv_layer.kernel,
          bias=bias,
          mu=self.bn_layer.moving_mean,
          var=self.bn_layer.moving_variance,
          gamma=self.bn_layer.gamma,
          beta=self.bn_layer.beta,
          epsilon=self.bn_layer.epsilon)

      self._quantize_weights(training)
      outputs = self._run_folded_conv(inputs, training)

    # Bias Add
    outputs = self._run_folded_bias_add(outputs)

    # Quantize activation
    for quantize_activation in self._quantize_activations:
      quantize_activation.training = training

    self.quantize_config.set_quantize_activations(self,
                                                  self._quantize_activations)

    if self.activation is not None:
      return self.activation(outputs)

    return outputs

  def get_config(self):
    base_config = super(VitisConvBNQuantize, self).get_config()
    config = {
        'conv_layer': serialize_keras_object(self.conv_layer),
        'bn_layer': serialize_keras_object(self.bn_layer),
        'activation': activations.serialize(self.activation),
        'freeze_bn_delay': self.freeze_bn_delay,
        'quantize_config': serialize_keras_object(self.quantize_config),
        'mode': self.mode
    }
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    config = config.copy()
    conv_layer = tf.keras.layers.deserialize(config.pop('conv_layer'))
    bn_layer = tf.keras.layers.deserialize(config.pop('bn_layer'))

    activation = config.pop('activation')
    quantize_config = deserialize_keras_object(
        config.pop('quantize_config'),
        module_objects=globals(),
        custom_objects=None)

    freeze_bn_delay = config.pop('freeze_bn_delay')
    mode = config.pop('mode')

    return cls(
        conv_layer=conv_layer,
        bn_layer=bn_layer,
        activation=activation,
        freeze_bn_delay=freeze_bn_delay,
        quantize_config=quantize_config,
        mode=mode,
        **config)

  @property
  def mode(self):
    return self._mode

  @mode.setter
  def mode(self, value):
    self._mode = value


@register_keras_serializable(package='Vitis', name='VitisDepthwiseConvBN')
class VitisDepthwiseConvBN(tf.keras.layers.Layer):
  """A Wrapper layer of folded convolution and batchnorm."""

  def __init__(self, conv_layer, bn_layer, activation=None, **kwargs):
    assert isinstance(conv_layer, tf.keras.layers.DepthwiseConv2D)
    assert isinstance(bn_layer, tf.keras.layers.BatchNormalization)
    #(TODO) check conv activation.
    self.conv_layer = conv_layer
    self.bn_layer = bn_layer
    self.activation = activations.get(activation)

    if 'name' not in kwargs:
      kwargs['name'] = self._make_layer_name(conv_layer, bn_layer)

    super(VitisDepthwiseConvBN, self).__init__(**kwargs)

    self._track_trackable(conv_layer, name='conv_layer')
    self._track_trackable(bn_layer, name='bn_layer')

  @staticmethod
  def _make_layer_name(conv_layer, bn_layer):
    layer_name = '{}_{}_{}'.format('quant', conv_layer.name, bn_layer.name)
    return layer_name

  def build(self, input_shape=None):
    super(VitisDepthwiseConvBN, self).build(input_shape)
    if not self.conv_layer.built or not self.bn_layer.built:
      self.conv_layer.build(input_shape)
      conv_out_shape = self.conv_layer.compute_output_shape(input_shape)
      self.bn_layer.build(conv_out_shape)
    self.built = True

  def call(self, inputs, training=None):
    if training is None:
      training = tf.keras.backend.learning_phase()

    outputs = self.conv_layer.call(inputs)
    outputs = self.bn_layer.call(outputs, training=training)
    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def get_config(self):
    base_config = super(VitisDepthwiseConvBN, self).get_config()
    config = {
        'conv_layer': serialize_keras_object(self.conv_layer),
        'bn_layer': serialize_keras_object(self.bn_layer),
        'activation': activations.serialize(self.activation),
    }
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    config = config.copy()
    conv_layer = tf.keras.layers.deserialize(config.pop('conv_layer'))
    bn_layer = tf.keras.layers.deserialize(config.pop('bn_layer'))
    activation = config.pop('activation')

    return cls(
        conv_layer=conv_layer,
        bn_layer=bn_layer,
        activation=activation,
        **config)


@register_keras_serializable(
    package='Vitis', name='VitisDepthwiseConvBNQuantize')
class VitisDepthwiseConvBNQuantize(tf.keras.layers.Layer):
  """A Wrapper layer emulate quantization of folded convolution and batchnorm."""

  def __init__(self, conv_layer, bn_layer, activation, freeze_bn_delay,
               quantize_config, mode, **kwargs):
    assert isinstance(conv_layer, tf.keras.layers.DepthwiseConv2D)
    assert isinstance(bn_layer, tf.keras.layers.BatchNormalization)
    #(TODO) check conv activation.
    self.conv_layer = conv_layer
    self.bn_layer = bn_layer
    self.activation = activations.get(activation)

    if 'name' not in kwargs:
      kwargs['name'] = self._make_layer_name(conv_layer, bn_layer)

    super(VitisDepthwiseConvBNQuantize, self).__init__(**kwargs)
    self.quantize_config = quantize_config
    self._mode = mode
    if freeze_bn_delay is not None:
      self.freeze_bn_delay = int(freeze_bn_delay)
    else:
      self.freeze_bn_delay = None

    self._track_trackable(conv_layer, name='conv_layer')
    self._track_trackable(bn_layer, name='bn_layer')

  @staticmethod
  def _make_layer_name(conv_layer, bn_layer):
    layer_name = '{}_{}_{}'.format('quant', conv_layer.name, bn_layer.name)
    return layer_name

  def _get_shape_map(self):
    return {
        'depthwise_kernel': self.conv_layer.depthwise_kernel.shape,
        'bias': self.bn_layer.beta.shape
    }

  def _build_for_quantization(self):
    """All Keras build() logic for quantization for fused layers."""
    self.optimizer_step = self.add_weight(
        'optimizer_step',
        initializer=keras.initializers.Constant(0),
        dtype=dtypes.int32,
        trainable=False)

    # Build weight quantizer variables with shape of conv layer kernel and bn layer gamma.
    shape_map = self._get_shape_map()
    quantizable_weights = self.quantize_config.get_quantizable_weights()
    weight_quantizers = self.quantize_config.get_weight_quantizers()

    self._weight_vars = []
    for weight, quantizer in zip(quantizable_weights, weight_quantizers):
      if weight not in shape_map:
        logger.error('Fail to get shape for {} of layer {}.'.format(
            weight, self))
      quantizer_vars = quantizer.build(shape_map[weight], weight, self)
      self._weight_vars.append((weight, quantizer, quantizer_vars))

    # Build activation quantizers
    self._quantize_activations = []
    for activation, quantizer in self.quantize_config.get_activations_and_quantizers(
        self):
      quantize_activation = vitis_quantize_aware_activation.QuantizeAwareActivation(
          activation, quantizer, self.mode, self.optimizer_step, self)
      self._quantize_activations.append(quantize_activation)

  def build(self, input_shape=None):
    super(VitisDepthwiseConvBNQuantize, self).build(input_shape)
    if not self.conv_layer.built or not self.bn_layer.built:
      self.conv_layer.build(input_shape)
      conv_out_shape = self.conv_layer.compute_output_shape(input_shape)
      self.bn_layer.build(conv_out_shape)

    # The folded conv always have bias
    self.use_bias = True
    self.built = True

    self._build_for_quantization()

  def _make_quantizer_fn(self, quantizer, x, training, mode, quantizer_vars):
    """Use currying to return True/False specialized fns to the cond."""

    def quantizer_fn():
      return quantizer(x, training, mode, weights=quantizer_vars)

    return quantizer_fn

  def _get_batch_mean_var(self, inputs):
    # Get batch_mean and variance, here we use the codes copy from:
    # https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/python/keras/layers/normalization.py#L806
    input_shape = inputs.shape
    ndims = len(input_shape)
    reduction_axes = [i for i in range(ndims) if i not in self.bn_layer.axis]
    keep_dims = self.bn_layer.virtual_batch_size is not None or len(
        self.bn_layer.axis) > 1
    return self.bn_layer._moments(
        math_ops.cast(inputs, self.bn_layer._param_dtype),
        reduction_axes,
        keep_dims=keep_dims)

  def _quantize_weights(self, training):
    # Quantize the folded kernel and bias
    for weight, quantizer, quantizer_vars in self._weight_vars:
      weight_tensor = getattr(self, weight)
      quantized_weight = tf_utils.smart_cond(
          training,
          self._make_quantizer_fn(quantizer, weight_tensor, True, self.mode,
                                  quantizer_vars),
          self._make_quantizer_fn(quantizer, weight_tensor, False, self.mode,
                                  quantizer_vars))
      setattr(self, weight, quantized_weight)

  def _run_folded_conv(self, inputs, training):
    """Run folded convolution.

       The folded conv do not has real variables so it can not be created by
       normal keras.Conv2D layers. Here we use the codes copy from
       https://github.com/tensorflow/tensorflow/blob/9edbe5075f79a4a95ed14a2be831f9b59e61f49d/tensorflow/python/keras/layers/convolutional.py#L2366
    """
    outputs = backend.depthwise_conv2d(
        inputs,
        self.depthwise_kernel,
        strides=self.conv_layer.strides,
        padding=self.conv_layer.padding,
        dilation_rate=self.conv_layer.dilation_rate,
        data_format=self.conv_layer.data_format)
    return outputs

  def _run_folded_bias_add(self, inputs):
    outputs = inputs
    # Bias add
    if self.use_bias:
      outputs = backend.bias_add(
          outputs, self.bias, data_format=self.conv_layer.data_format)
    return outputs

  def call(self, inputs, training=None):
    if training is None:
      training = tf.keras.backend.learning_phase()

    bias = tf_utils.smart_cond(self.conv_layer.use_bias,
                               lambda: self.conv_layer.bias, lambda: 0)

    if training:
      self.optimizer_step.assign_add(1)

    freeze_bn = tf_utils.smart_cond(
        self.freeze_bn_delay is not None, lambda: math_ops.greater_equal(
            self.optimizer_step, self.freeze_bn_delay), lambda: False)
    #  tf.print('step: {}, freeze_bn: {}'.format(self.optimizer_step, freeze_bn))

    if training and not freeze_bn:
      # Run float conv and bn to update the moving mean and variance
      conv_out = self.conv_layer.call(inputs)
      bn_out = self.bn_layer.call(conv_out, training=training)

      mu_bt, var_bt = self._get_batch_mean_var(conv_out)
      sigma_bt = math_ops.rsqrt(var_bt + self.bn_layer.epsilon)

      # Get folded depthwise_kernel and bias
      self.depthwise_kernel, self.bias = _get_folded_kernel_bias(
          conv_type='DepthwiseConv2D',
          kernel=self.conv_layer.depthwise_kernel,
          bias=bias,
          mu=mu_bt,
          var=var_bt,
          gamma=self.bn_layer.gamma,
          beta=self.bn_layer.beta,
          epsilon=self.bn_layer.epsilon)

      # BatchNorm Correction
      corr_scale, corr_recip, corr_offset = _get_bn_correction(
          conv_type='DepthwiseConv2D',
          kernel=self.conv_layer.depthwise_kernel,
          bias=bias,
          mu_bt=mu_bt,
          var_bt=var_bt,
          mu_mv=self.bn_layer.moving_mean,
          var_mv=self.bn_layer.moving_variance,
          gamma=self.bn_layer.gamma,
          epsilon=self.bn_layer.epsilon)

      self.depthwise_kernel = math_ops.mul(self.depthwise_kernel, corr_scale)
      self.bias = math_ops.add(self.bias, corr_offset)

      self._quantize_weights(training)
      outputs = self._run_folded_conv(inputs, training)
      # BatchNorm Correction for convolution outputs
      outputs = math_ops.mul(outputs, corr_recip)
    else:
      self.depthwise_kernel, self.bias = _get_folded_kernel_bias(
          conv_type='DepthwiseConv2D',
          kernel=self.conv_layer.depthwise_kernel,
          bias=bias,
          mu=self.bn_layer.moving_mean,
          var=self.bn_layer.moving_variance,
          gamma=self.bn_layer.gamma,
          beta=self.bn_layer.beta,
          epsilon=self.bn_layer.epsilon)

      self._quantize_weights(training)
      outputs = self._run_folded_conv(inputs, training)

    # Bias Add
    outputs = self._run_folded_bias_add(outputs)

    # Quantize activation
    for quantize_activation in self._quantize_activations:
      quantize_activation.training = training

    self.quantize_config.set_quantize_activations(self,
                                                  self._quantize_activations)

    if self.activation is not None:
      return self.activation(outputs)

    return outputs

  def get_config(self):
    base_config = super(VitisDepthwiseConvBNQuantize, self).get_config()
    config = {
        'conv_layer': serialize_keras_object(self.conv_layer),
        'bn_layer': serialize_keras_object(self.bn_layer),
        'activation': activations.serialize(self.activation),
        'freeze_bn_delay': self.freeze_bn_delay,
        'quantize_config': serialize_keras_object(self.quantize_config),
        'mode': self.mode
    }
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    config = config.copy()
    conv_layer = tf.keras.layers.deserialize(config.pop('conv_layer'))
    bn_layer = tf.keras.layers.deserialize(config.pop('bn_layer'))

    activation = config.pop('activation')
    quantize_config = deserialize_keras_object(
        config.pop('quantize_config'),
        module_objects=globals(),
        custom_objects=None)

    freeze_bn_delay = config.pop('freeze_bn_delay')
    mode = config.pop('mode')

    return cls(
        conv_layer=conv_layer,
        bn_layer=bn_layer,
        activation=activation,
        freeze_bn_delay=freeze_bn_delay,
        quantize_config=quantize_config,
        mode=mode,
        **config)

  @property
  def mode(self):
    return self._mode

  @mode.setter
  def mode(self, value):
    self._mode = value
