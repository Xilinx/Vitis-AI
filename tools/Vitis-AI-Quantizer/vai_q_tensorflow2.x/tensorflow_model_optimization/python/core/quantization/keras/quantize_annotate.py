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
"""Quantize Annotate Wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

deserialize_keras_object = tf.keras.utils.deserialize_keras_object
serialize_keras_object = tf.keras.utils.serialize_keras_object


class QuantizeAnnotate(tf.keras.layers.Wrapper):
  """Annotates layers which quantization should be applied to.

  QuantizeAnnotate does not actually apply quantization to the underlying
  layers but acts as a way to specify which layers quantization should be
  applied to.

  The wrapper functions as a NoOp or pass-through wrapper by simply delegating
  calls to the underlying layer. The presence of this wrapper indicates to code
  which actually applies quantization to determine which layers should be
  modified.
  """

  _UNSUPPORTED_LAYER_ERROR_MSG = (
      'Layer {} not supported for quantization. Layer should either inherit '
      'QuantizeEmulatableLayer or be a supported keras built-in layer.')

  def __init__(self, layer, quantize_config=None, **kwargs):
    """Create a quantize annotate wrapper over a keras layer.

    Args:
      layer: The keras layer to be quantized.
      quantize_config: Optional `QuantizeConfig` to quantize the layer.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
    """
    super(QuantizeAnnotate, self).__init__(layer, **kwargs)

    if layer is None:
      raise ValueError('`layer` cannot be None.')

    # Check against keras.Model since it is an instance of keras.layers.Layer.
    if not isinstance(layer, tf.keras.layers.Layer) or isinstance(
        layer, tf.keras.Model):
      raise ValueError(
          '`layer` can only be a `tf.keras.layers.Layer` instance. '
          'You passed an instance of type: {input}.'.format(
              input=layer.__class__.__name__))

    self.quantize_config = quantize_config

    self._track_trackable(layer, name='layer')
    # Enables end-user to annotate the first layer in Sequential models, while
    # passing the input shape to the original layer.
    #
    # tf.keras.Sequential(
    #   quantize_annotate_layer(tf.keras.layers.Dense(2, input_shape=(3,)))
    # )
    #
    # as opposed to
    #
    # tf.keras.Sequential(
    #   quantize_annotate_layer(tf.keras.layers.Dense(2), input_shape=(3,))
    # )
    #
    # Without this code, the QuantizeAnnotate wrapper doesn't have an input
    # shape and being the first layer, this causes the model to not be
    # built. Being not built is confusing since the end-user has passed an
    # input shape.
    if (not hasattr(self, '_batch_input_shape') and
        hasattr(layer, '_batch_input_shape')):
      self._batch_input_shape = self.layer._batch_input_shape  # pylint: disable=protected-access

  def call(self, inputs, training=None):
    return self.layer.call(inputs)

  def get_config(self):
    base_config = super(QuantizeAnnotate, self).get_config()
    config = {'quantize_config': serialize_keras_object(self.quantize_config)}
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    config = config.copy()

    quantize_config = deserialize_keras_object(
        config.pop('quantize_config'),
        module_objects=globals(),
        custom_objects=None)

    layer = tf.keras.layers.deserialize(config.pop('layer'))

    return cls(layer=layer, quantize_config=quantize_config, **config)

  def compute_output_shape(self, input_shape):
    return self.layer.compute_output_shape(input_shape)

  @property
  def trainable(self):
    return self.layer.trainable

  @trainable.setter
  def trainable(self, value):
    self.layer.trainable = value

  @property
  def trainable_weights(self):
    return self.layer.trainable_weights

  @property
  def non_trainable_weights(self):
    return self.layer.non_trainable_weights + self._non_trainable_weights

  @property
  def updates(self):
    return self.layer.updates + self._updates

  @property
  def losses(self):
    return self.layer.losses + self._losses

  def get_weights(self):
    return self.layer.get_weights()

  def set_weights(self, weights):
    self.layer.set_weights(weights)
