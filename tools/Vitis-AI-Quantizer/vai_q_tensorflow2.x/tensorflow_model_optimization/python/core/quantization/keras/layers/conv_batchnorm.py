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
"""Convolution with folded batch normalization layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.keras.layers import convolutional
from tensorflow.python.keras.layers import deserialize as deserialize_layer
from tensorflow.python.keras.layers import normalization
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops

from tensorflow_model_optimization.python.core.quantization.keras import quantizers
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantizers

keras = tf.keras


class _ConvBatchNormMixin(object):
  """Provides shared functionality between fused batchnorm layers."""

  def _build_for_quantization(self):
    """All Keras build() logic for quantization for fused layers."""
    if not self.is_quantized:
      return

    self._weight_quantizer_vars = self.weight_quantizer.build(
        self.weights[0].shape, 'weight', self)

    self.optimizer_step = self.add_weight(
        'optimizer_step',
        initializer=initializers.Constant(-1),
        dtype=dtypes.int32,
        trainable=False)

    # TODO(alanchiao): re-explore if we can handle this with
    # QuantizeAwareActivation.
    self._activation_min_var = self.add_variable(  # pylint: disable=protected-access
        'activation_min',
        initializer=initializers.Constant(-6.0),
        trainable=False)
    self._activation_max_var = self.add_variable(  # pylint: disable=protected-access
        'activation_max',
        initializer=initializers.Constant(6.0),
        trainable=False)

  def _apply_weight_quantizer(self, training, folded_conv_kernel):
    """All Keras call() logic for applying weight quantization."""

    def make_quantizer_fn(training):
      """Return quantizer conditioned on whether training or not."""

      def quantizer_fn():
        return self.weight_quantizer(
            folded_conv_kernel,
            training,
            weights=self._weight_quantizer_vars)  # pylint: disable=protected-access

      return quantizer_fn

    return tf_utils.smart_cond(training, make_quantizer_fn(True),
                               make_quantizer_fn(False))

  def _apply_activation_quantizer(self, training, activation_output):
    """All Keras call() logic for applying weight quantization."""

    def make_quantizer_fn(training):
      """Return quantizer conditioned on whether training or not."""

      def quantizer_fn():
        weights = {
            'min_var': self._activation_min_var,  # pylint: disable=protected-access
            'max_var': self._activation_max_var}  # pylint: disable=protected-access
        return self.activation_quantizer(
            activation_output,
            training,
            weights=weights)

      return quantizer_fn

    return tf_utils.smart_cond(training, make_quantizer_fn(True),
                               make_quantizer_fn(False))

  @staticmethod
  def _from_config(cls_initializer, config):
    """All shared from_config logic for fused layers."""
    config = config.copy()
    # use_bias is not an argument of this class, as explained by
    # comment in __init__.
    config.pop('use_bias')
    is_advanced_activation = 'class_name' in config['post_activation']
    if is_advanced_activation:
      config['post_activation'] = deserialize_layer(config['post_activation'])
    else:
      config['post_activation'] = activations.deserialize(
          config['post_activation'])

    return cls_initializer(**config)

  def _get_config(self, conv_config):
    """All shared get_config logic for fused layers."""
    batchnorm_config = self.batchnorm.get_config()

    # Both BatchNorm and Conv2D have config items from base layer. Since
    # _ConvBatchNorm2D inherits from Conv2D, we should use base layer config
    # items from self, rather than self.batchnorm.
    # For now, deleting 'name', but ideally all base_config items should be
    # removed.
    # TODO(pulkitb): Raise error if base_configs in both layers incompatible.
    batchnorm_config.pop('name')

    is_advanced_activation = isinstance(self.post_activation,
                                        keras.layers.Layer)
    if is_advanced_activation:
      serialized_activation = keras.utils.serialize_keras_object(
          self.post_activation)
    else:
      serialized_activation = activations.serialize(self.post_activation)
    config = {
        'is_quantized': self.is_quantized,
        'post_activation': serialized_activation
    }

    return dict(
        list(conv_config.items()) + list(batchnorm_config.items()) +
        list(config.items()))


class _ConvBatchNorm2D(_ConvBatchNormMixin, convolutional.Conv2D):
  """Layer for emulating the folding of batch normalization into Conv during serving.

  Implements the emulation, as described in https://arxiv.org/abs/1712.05877.
  Note that in the
  emulated form, there are two convolutions for each convolution in the original
  model.

  Notably, this layer adds the quantization ops  itself, instead of relying on
  the wrapper. The reason is that the weight (folded_conv_kernel) is an
  intermediate tensor instead of a variable tensor, and therefore not accessible
  to the wrapper at build() time.
  """

  # TODO(alanchiao): remove these defaults since in practice,
  # they will be provided by the unfolded layers.
  #
  # Note: the following are not parameters even though they are for Conv2D.
  # 1. use_bias. This is because a Conv2D bias would be redundant with
  # BatchNormalization's bias.
  # 2. activation. We can only mathematically fold through linear operations,
  # so an activation in the Conv2D prevents batchnorm folding.
  def __init__(
      self,
      # Conv2D params
      filters,
      kernel_size,
      strides=(1, 1),
      padding='valid',
      data_format=None,
      dilation_rate=(1, 1),
      kernel_initializer='glorot_uniform',
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      name=None,
      # BatchNormalization params
      axis=-1,
      momentum=0.99,
      epsilon=1e-3,
      center=True,
      scale=True,
      beta_initializer='zeros',
      gamma_initializer='ones',
      moving_mean_initializer='zeros',
      moving_variance_initializer='ones',
      beta_regularizer=None,
      gamma_regularizer=None,
      beta_constraint=None,
      gamma_constraint=None,
      renorm=False,
      renorm_clipping=None,
      renorm_momentum=0.99,
      fused=None,
      trainable=True,
      virtual_batch_size=None,
      adjustment=None,
      # Post-batchnorm activation.
      post_activation=None,
      # quantization params
      is_quantized=True,
      **kwargs):
    super(_ConvBatchNorm2D, self).__init__(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        use_bias=False,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        name=name,
        **kwargs)

    self.batchnorm = normalization.BatchNormalization(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint,
        renorm=renorm,
        renorm_clipping=renorm_clipping,
        renorm_momentum=renorm_momentum,
        fused=fused,
        trainable=trainable,
        virtual_batch_size=virtual_batch_size,
        adjustment=adjustment,
    )

    # Named as post_activation to not conflict with Layer self.activation.
    self.post_activation = activations.get(post_activation)

    self.is_quantized = is_quantized
    if self.is_quantized:
      self.weight_quantizer = default_8bit_quantizers.Default8BitConvWeightsQuantizer(
      )

      self.activation_quantizer = quantizers.MovingAverageQuantizer(
          num_bits=8, per_axis=False, symmetric=False, narrow_range=False)

  def build(self, input_shape):
    # responsible for trainable self.kernel weights
    super(_ConvBatchNorm2D, self).build(input_shape)

    # resposible for trainable gamma and beta weights
    self.batchnorm.build(self.compute_output_shape(input_shape))

    self._build_for_quantization()

  def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase()

    conv_out = super(_ConvBatchNorm2D, self).call(inputs)

    # Not all the computations in the batchnorm need to happen,
    # but this avoids duplicating code (e.g. moving_average).
    self.batchnorm.call(conv_out)

    folded_conv_kernel_multiplier = self.batchnorm.gamma * math_ops.rsqrt(
        self.batchnorm.moving_variance + self.batchnorm.epsilon)
    folded_conv_kernel = math_ops.mul(
        folded_conv_kernel_multiplier, self.kernel, name='folded_conv_kernel')

    folded_conv_bias = math_ops.subtract(
        self.batchnorm.beta,
        self.batchnorm.moving_mean * folded_conv_kernel_multiplier,
        name='folded_conv_bias')

    if self.is_quantized:
      folded_conv_kernel = self._apply_weight_quantizer(training,
                                                        folded_conv_kernel)

    # Second convolution doesn't need new trainable weights, so we
    # cannot reuse Conv2D layer.
    # TODO(alanchiao):
    # 1. See if we can at least reuse the bias logic.
    # 2. See if we need to fork between conv2d and conv2d_v2 for
    #    TensorFlow 1.XX and 2.XX.

    # Taken from keras/layers/convolutional.py:183
    if self.padding == 'causal':
      op_padding = 'valid'
    else:
      op_padding = self.padding
    if not isinstance(op_padding, (list, tuple)):
      op_padding = op_padding.upper()

    folded_conv_out = nn_ops.conv2d(
        inputs,
        folded_conv_kernel,
        strides=self.strides,
        padding=op_padding,
        data_format=conv_utils.convert_data_format(self.data_format,
                                                   self.rank + 2),
        dilations=self.dilation_rate,
        name='folded_conv_out',
    )

    # Taken from keras/layers/convolutional.py:200
    if self.data_format == 'channels_first':
      if self.rank == 1:
        # nn.bias_add does not accept a 1D input tensor.
        bias = array_ops.reshape(folded_conv_bias, (1, self.filters, 1))
        folded_conv_out += bias
      else:
        outputs = nn.bias_add(
            folded_conv_out, folded_conv_bias, data_format='NCHW')
    else:
      outputs = nn.bias_add(
          folded_conv_out, folded_conv_bias, data_format='NHWC')

    if self.post_activation is not None:
      outputs = self.post_activation(outputs)
    if self.is_quantized:
      outputs = self._apply_activation_quantizer(training, outputs)
    return outputs

  def get_config(self):
    conv_config = super(_ConvBatchNorm2D, self).get_config()
    return self._get_config(conv_config)

  @classmethod
  def from_config(cls, config):
    return _ConvBatchNormMixin._from_config(cls, config)


class _DepthwiseConvBatchNorm2D(_ConvBatchNormMixin,
                                convolutional.DepthwiseConv2D):
  """Layer for emulating the folding of batch normalization into DepthwiseConv during serving.

  See ConvBatchNorm2D for detailed comments.
  """

  def __init__(
      self,
      # DepthwiseConv2D params
      kernel_size,
      strides=(1, 1),
      padding='valid',
      depth_multiplier=1,
      data_format=None,
      depthwise_initializer='glorot_uniform',
      depthwise_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      depthwise_constraint=None,
      bias_constraint=None,
      name=None,
      # BatchNormalization params
      axis=-1,
      momentum=0.99,
      epsilon=1e-3,
      center=True,
      scale=True,
      beta_initializer='zeros',
      gamma_initializer='ones',
      moving_mean_initializer='zeros',
      moving_variance_initializer='ones',
      beta_regularizer=None,
      gamma_regularizer=None,
      beta_constraint=None,
      gamma_constraint=None,
      renorm=False,
      renorm_clipping=None,
      renorm_momentum=0.99,
      fused=None,
      trainable=True,
      virtual_batch_size=None,
      adjustment=None,
      # Post-batchnorm activation instance.
      post_activation=None,
      # quantization params
      is_quantized=True,
      **kwargs):
    super(_DepthwiseConvBatchNorm2D, self).__init__(
        kernel_size,
        strides=strides,
        padding=padding,
        depth_multiplier=depth_multiplier,
        data_format=data_format,
        use_bias=False,
        depthwise_initializer=depthwise_initializer,
        depthwise_regularizer=depthwise_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        depthwise_constraint=depthwise_constraint,
        bias_constraint=bias_constraint,
        name=name,
        **kwargs)

    self.batchnorm = normalization.BatchNormalization(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint,
        renorm=renorm,
        renorm_clipping=renorm_clipping,
        renorm_momentum=renorm_momentum,
        fused=fused,
        trainable=trainable,
        virtual_batch_size=virtual_batch_size,
        adjustment=adjustment,
    )
    self.post_activation = activations.get(post_activation)

    self.is_quantized = is_quantized
    if self.is_quantized:
      self.weight_quantizer = default_8bit_quantizers.Default8BitConvWeightsQuantizer(
      )

      self.activation_quantizer = quantizers.MovingAverageQuantizer(
          num_bits=8, per_axis=False, symmetric=False, narrow_range=False)

  def build(self, input_shape):
    # responsible for trainable self.kernel weights
    super(_DepthwiseConvBatchNorm2D, self).build(input_shape)

    # resposible for trainable gamma and beta weights
    self.batchnorm.build(self.compute_output_shape(input_shape))

    self._build_for_quantization()

  def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase()

    conv_out = super(_DepthwiseConvBatchNorm2D, self).call(inputs)

    self.batchnorm.call(conv_out)

    folded_conv_kernel_multiplier = self.batchnorm.gamma * math_ops.rsqrt(
        self.batchnorm.moving_variance + self.batchnorm.epsilon)

    folded_conv_bias = math_ops.subtract(
        self.batchnorm.beta,
        self.batchnorm.moving_mean * folded_conv_kernel_multiplier,
        name='folded_conv_bias')

    depthwise_weights_shape = [
        self.depthwise_kernel.get_shape().as_list()[2],
        self.depthwise_kernel.get_shape().as_list()[3]
    ]
    folded_conv_kernel_multiplier = array_ops.reshape(
        folded_conv_kernel_multiplier, depthwise_weights_shape)

    folded_conv_kernel = math_ops.mul(
        folded_conv_kernel_multiplier,
        self.depthwise_kernel,
        name='folded_conv_kernel')

    if self.is_quantized:
      folded_conv_kernel = self._apply_weight_quantizer(training,
                                                        folded_conv_kernel)

    # TODO(alanchiao): this is an internal API.
    # See if Keras would make this public, like
    # backend.conv2d is.
    #
    # From DepthwiseConv2D layer call() function.
    folded_conv_out = K.depthwise_conv2d(
        inputs,
        folded_conv_kernel,
        strides=self.strides,
        padding=self.padding,
        dilation_rate=self.dilation_rate,
        data_format=self.data_format,
    )

    outputs = K.bias_add(
        folded_conv_out, folded_conv_bias, data_format=self.data_format)

    if self.post_activation is not None:
      outputs = self.post_activation(outputs)
    if self.is_quantized:
      outputs = self._apply_activation_quantizer(training, outputs)
    return outputs

  def get_config(self):
    conv_config = super(_DepthwiseConvBatchNorm2D, self).get_config()
    return self._get_config(conv_config)

  @classmethod
  def from_config(cls, config):
    return _ConvBatchNormMixin._from_config(cls, config)
