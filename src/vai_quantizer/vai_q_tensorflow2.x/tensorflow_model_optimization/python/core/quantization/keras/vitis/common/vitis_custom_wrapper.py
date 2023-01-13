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
"""Wrapper which is custom layer over underlying layer.

   `CustomOpWrapper` is responsible for modifying the construction of the
   underlying layer to ensure proper attributes are placed in the
   graph.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import copy

from tensorflow.python.util import tf_inspect
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

register_keras_serializable = tf.keras.utils.register_keras_serializable
deserialize_keras_object = tf.keras.utils.deserialize_keras_object
serialize_keras_object = tf.keras.utils.serialize_keras_object
logger = common_utils.VAILogger


@register_keras_serializable(package='Vitis', name='CustomOpWrapper')
class CustomOpWrapper(tf.keras.layers.Wrapper):
  """Mark this layer as a custom layer and set some attributes"""

  def __init__(self, layer, **kwargs):
    """Create a custom layer wrapper for a keras layer.

    Args:
      layer: The keras layer to be quantized.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
    """
    if layer is None:
      logger.error('`layer` cannot be None.')

    # Check against keras.Model since it is an instance of keras.layers.Layer.
    if not isinstance(layer, tf.keras.layers.Layer) or isinstance(
        layer, tf.keras.Model):
      logger.error(
                   '[Quantizer_TF2_Unsupported_Layer][Unsupported layer type]'
                   '`layer` can only be a `tf.keras.layers.Layer` instance. '
                   'You passed an instance of type: {input}.'.format(
                       input=layer.__class__.__name__))
    if 'name' not in kwargs:
      kwargs['name'] = layer.name

    super(CustomOpWrapper, self).__init__(layer, **kwargs)

    self._track_trackable(layer, name='layer')

  def build(self, input_shape):
    super(CustomOpWrapper, self).build(input_shape)

  def compute_output_shape(self, input_shape):
    return self.layer.compute_output_shape(self.layer.input_shape)

  def call(self, inputs, training=None):
    args = tf_inspect.getfullargspec(self.layer.call).args
    if 'training' in args:
      outputs = self.layer.call(inputs, training=training)
    else:
      outputs = self.layer.call(inputs)
    return outputs

  def get_config(self):
    base_config = super(CustomOpWrapper, self).get_config()
    config = {}
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    config = config.copy()

    layer = tf.keras.layers.deserialize(config.pop('layer'))

    return cls(layer=layer, **config)

  @property
  def trainable_weights(self):
    return self.layer.trainable_weights

  @property
  def non_trainable_weights(self):
    return self.layer.non_trainable_weights

  @property
  def updates(self):
    return self.layer.updates + self._updates

  @property
  def losses(self):
    return self.layer.losses + self._losses


_types_dict = {"CustomOpWrapper", CustomOpWrapper}
