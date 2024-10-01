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

import math
import string

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
import distutils

from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine.base_layer import Layer
from keras.utils import tf_utils

from tensorflow_model_optimization.python.core.quantization.keras.vitis.base import quantizer as quantizer_mod
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils



__all__ = [
    'VitisMultiHeadAttentionQuantize'
]

activations = tf.keras.activations
backend = tf.keras.backend
register_keras_serializable = tf.keras.utils.register_keras_serializable
serialize_keras_object = tf.keras.utils.serialize_keras_object
deserialize_keras_object = tf.keras.utils.deserialize_keras_object
logger = common_utils.VAILogger

_CHR_IDX = string.ascii_lowercase

def _compute_attention_mask(query, value, key=None, attention_mask=None, use_causal_mask=False):
  query_mask = getattr(query, '_keras_mask', None)
  value_mask = getattr(value, '_keras_mask', None)
  key_mask = getattr(key, '_keras_mask', None)
  auto_mask = None

  if query_mask is not None:
    query_mask = tf.cast(query_mask, tf.bool)
    auto_mask = query_mask[:, :, tf.newaxis]
  if value_mask is not None:
    value_mask = tf.cast(value_mask, tf.bool)
    mask = value_mask[:, tf.newaxis, :]
    auto_mask = mask if auto_mask is None else auto_mask & mask
  if key_mask is not None:
    key_mask = tf.cast(key_mask, tf.bool)
    mask = key_mask[:, tf.newaxis, :]
    auto_mask = mask if auto_mask is None else auto_mask & mask
  if use_causal_mask:
    mask = _compute_causal_mask(query, key)
    auto_mask = mask if auto_mask is None else auto_mask & mask
  if auto_mask is not None:
    attention_mask = (auto_mask if attention_mask is None else tf.cast(attention_mask, bool) & auto_mask)
  return attention_mask

def _compute_causal_mask(query, value=None):
  q_seq_length = tf.shape(query)[1]
  v_seq_length = q_seq_length if value is None else tf.shape(value)[1]
  return tf.linalg.band_part(tf.ones((1, q_seq_length, v_seq_length), tf.bool), -1, 0)

@register_keras_serializable(package='Vitis', name='VitisMultiHeadAttentionQuantize')
class VitisMultiHeadAttentionQuantize(tf.keras.layers.Layer):
  """"""

  def __init__(self, layer, quantize_config, **kwargs):
    #super().__init__(**kwargs)
    self.num_heads = layer._num_heads
    self.key_dim = layer._key_dim
    self.key_shape = layer._key_shape
    self.value_shape = layer._value_shape
    self.query_shape = layer._query_shape
    self.attention_axes = layer._attention_axes
    self.dropout = layer._dropout
    self.use_bias = layer._use_bias
    self.kernel_initializer = initializers.get(layer._kernel_initializer)
    self.bias_initializer = initializers.get(layer._bias_initializer)
    self.kernel_regularizer = regularizers.get(layer._kernel_regularizer)
    self.bias_regularizer = regularizers.get(layer._bias_regularizer)
    self.activity_regularizer = regularizers.get(layer._activity_regularizer)
    self.kernel_constraint = constraints.get(layer._kernel_constraint)
    self.bias_constraint = constraints.get(layer._bias_constraint)
    self.quantize_config = quantize_config
    self.value_dim = layer._value_dim
    self.built_from_signature = False
    self.built_for_quantization = False
    self.mode = "QCB"
    self.layer = layer

    if layer is None:
      logger.error('`layer` cannot be None.')

    # Check against keras.Model since it is an instance of keras.layers.Layer.
    if not isinstance(layer, tf.keras.layers.Layer) or isinstance(
        layer, tf.keras.Model):
      logger.error('`layer` can only be a `tf.keras.layers.Layer` instance. '
                   'You passed an instance of type: {input}.'.format(
                       input=layer.__class__.__name__))

    if quantize_config is None:
      logger.error('quantize_config cannot be None. It is needed to '
                   'quantize a layer.')

    if 'name' not in kwargs:
      kwargs['name'] = self._make_layer_name(self.layer)

    super(VitisMultiHeadAttentionQuantize, self).__init__(**kwargs)

    self._track_trackable(self.layer, name='quant_multi_head_attention_layer')

  @staticmethod
  def _make_layer_name(layer):
    return '{}_{}'.format('quant', layer.name)

  def _get_common_kwargs_for_sublayer(self):
    common_kwargs = dict(
      kernel_regularizer=self.kernel_regularizer,
      bias_regularizer=self.bias_regularizer,
      activity_regularizer=self.activity_regularizer,
      kernel_constraint=self.kernel_constraint,
      bias_constraint=self.bias_constraint,
    )
    # Create new clone of kernel/bias initializer, so that we don't reuse
    # the initializer instance, which could lead to same init value since
    # initializer is stateless.
    kernel_initializer = self.kernel_initializer.__class__.from_config(
      self.kernel_initializer.get_config()
    )
    bias_initializer = self.bias_initializer.__class__.from_config(
      self.bias_initializer.get_config()
    )
    common_kwargs["kernel_initializer"] = kernel_initializer
    common_kwargs["bias_initializer"] = bias_initializer
    return common_kwargs

  def _make_quantizer_fn(self, quantizer, x, training, mode, quantizer_vars):
    """Use currying to return True/False specialized fns to the cond."""

    def quantizer_fn():
      return quantizer(x, training, mode, weights=quantizer_vars)

    return quantizer_fn

  def _build_for_quantization(self):
  
    input_shapes = [self.query_shape, self.value_shape, self.key_shape]
    self._input_quantizer_vars = []
    for input_id, quantizer in self.quantize_config.get_input_and_quantizers(self.layer):
      quantizer_vars = quantizer.build(input_shapes[input_id], 'input_'+str(input_id), self)
      self._input_quantizer_vars.append((input_id, quantizer, quantizer_vars))

    self._weight_vars = []
    self._bias_vars = []
    if self.layer.weights:
      for weight, quantizer in self.get_weights_and_quantizers():
        if weight is None:
          continue

        quantizer_vars = quantizer.build(weight.shape, weight.name, self)
        self._weight_vars.append((weight, quantizer, quantizer_vars))
        # Need to ensure unquantized weights get trained as part of the wrapper. why???
        self._trainable_weights.append(weight)

      for bias, quantizer in self.quantize_config.get_biases_and_quantizers(self.layer):
        if bias is None:
          continue

        quantizer_vars = quantizer.build(bias.shape, bias.name, self)
        self._bias_vars.append((bias, quantizer, quantizer_vars))
        self._trainable_weights.append(bias)
    
    self._quantize_activations = []
    #for activation, quantizer in self.quantize_config.get_activations_and_quantizers(
    #    self):
    #  quantize_activation = vitis_quantize_aware_activation.QuantizeAwareActivation(
    #      activation, quantizer, 'QCB', self.optimizer_step, self._softmax)
    #  self._quantize_activations.append(quantize_activation)
    self.optimizer_step = self.layer.add_weight(
        'optimizer_step',
        initializer=keras.initializers.Constant(0),
        dtype=dtypes.int32,
        trainable=False)
    

  def _build_from_signature(self, query, value, key=None):
    self.layer._build_from_signature(query=query, value=value, key=key)
    self._build_for_quantization()

    self.built_from_signature = True

  def get_weights_and_quantizers(self):
    res = []
    quantizable_weights = self.quantize_config.get_quantizable_weights()
    weight_quantizers = self.quantize_config.get_weight_quantizers()
    layers = []
    for i in range(len(quantizable_weights)):
      layers.append(self.layer)
    for layer, weight, quantizer in zip(layers, quantizable_weights, weight_quantizers):
      res.append((eval('.'.join(str(i) for i in ['layer', weight])), quantizer))
    return res

  def _set_quantize_weights(self, quantize_weights):
    layers = []
    for i in range(len(quantize_weights)):
      layers.append(self.layer)
    quantizable_weights = self.quantize_config.get_quantizable_weights()
    for layer, weight, quantize_weight in zip(layers, quantizable_weights, quantize_weights):
      current_weight = eval('layer.'+weight)
      if current_weight.shape != quantize_weight.shape:
        logger.error('Existing layer weight shape {} is incompatible with '
                     'provided quantize weight shape {}'.format(
                         current_weight.shape, quantize_weight.shape))

      l = weight.split('.')
      setattr(eval('layer.'+'.'.join(i for i in l[:-1])), l[-1], quantize_weight)

  def call(self, query, value, key=None, attention_mask=None, return_attention_scores=False, training=None, use_causal_mask=False):

    if not self.built_for_quantization and tf.executing_eagerly():
      self._build_for_quantization()
      self.built_for_quantization = True

    if training is None:
      training = tf.keras.backend.learning_phase()

    attention_mask = _compute_attention_mask(query, value, key=key, attention_mask=attention_mask, use_causal_mask=use_causal_mask)
    if key is None:
      key = value
    if not self.built_from_signature:
      self._build_from_signature(query=query, value=value, key=key)
 
    inputs = [query, value, key]
    for input_id, input_quantizer, input_quantizer_vars in self._input_quantizer_vars:
      inputs[input_id] = common_utils.smart_cond(training, 
        self._make_quantizer_fn(input_quantizer, inputs[input_id], True, self.mode, input_quantizer_vars),
        self._make_quantizer_fn(input_quantizer, inputs[input_id], False, self.mode, input_quantizer_vars)
        )

    quantized_weights = []
    for unquantized_weight, quantizer, quantizer_vars in self._weight_vars:
      quantized_weight = common_utils.smart_cond(training,
        self._make_quantizer_fn(quantizer, unquantized_weight, True, self.mode, quantizer_vars),
        self._make_quantizer_fn(quantizer, unquantized_weight, False, self.mode, quantizer_vars)
        )
      quantized_weights.append(quantized_weight)
    self._set_quantize_weights(quantized_weights)
    query, key, value = inputs
    query_is_ragged = isinstance(query, tf.RaggedTensor)
    if query_is_ragged:
      query_lengths = query.nested_row_lengths()
      query = query.to_tensor()

    key_is_ragged = isinstance(key, tf.RaggedTensor)
    value_is_ragged = isinstance(value, tf.RaggedTensor)
    if key_is_ragged and value_is_ragged:
      bounding_shape = tf.math.maximum(key.bounding_shape(), value.bounding_shape())
      key = key.to_tensor(shape=bounding_shape)
      value = value.to_tensor(shape=bounding_shape)
    elif key_is_ragged:
      key = key.to_tensor(shape=tf.shape(value))
    elif value_is_ragged:
      value = value.to_tensor(shape=tf.shape(key))

    query = self.layer._query_dense(query)
    key = self.layer._key_dense(key)
    value = self.layer._value_dense(value)

    attention_output, attention_scores = self.layer._compute_attention(query, key, value, attention_mask, training)
    attention_output = self.layer._output_dense(attention_output)

    if query_is_ragged:
      attention_output = tf.RaggedTensor.from_tensor(attention_output, lengths=query_lengths)

    if return_attention_scores:
      return attention_output, attention_scores

    return attention_output

  @property
  def trainable_weights(self):
    return self._trainable_weights

  @property
  def weights(self):
    return self.layer.weights

  def get_config(self):
    base_config = super(VitisMultiHeadAttentionQuantize, self).get_config()
    config = {
      'layer': tf.keras.layers.serialize(self.layer),
      'quantize_config': serialize_keras_object(self.quantize_config)
    }
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    config = config.copy()
    layer = tf.keras.layers.deserialize(config.pop('layer'))
    quantize_config = deserialize_keras_object(
      config.pop('quantize_config'),
      module_objects=globals(),
      custom_objects=None)
    return cls(layer=layer, quantize_config=quantize_config, **config)

