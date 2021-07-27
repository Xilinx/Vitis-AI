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
"""Wrapper which applies quantization operations over underlying layer.

   `QuantizeWrapper` is responsible for modifying the construction of the
   underlying layer to ensure proper quantization operations are placed in the
   graph.

   These operations ensure proper introduction of inference time losses during
   training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import copy

# TODO(b/139939526): move to public API.
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util import tf_inspect
from tensorflow.python.keras.utils.generic_utils import register_keras_serializable
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

deserialize_keras_object = tf.keras.utils.deserialize_keras_object
serialize_keras_object = tf.keras.utils.serialize_keras_object
logger = common_utils.VAILogger


@register_keras_serializable(package='Vitis', name='QuantizeWrapper')
class QuantizeWrapper(tf.keras.layers.Wrapper):
  """Quantizes the weights and activations of the keras layer it wraps."""

  def __init__(self, layer, quantize_config, mode, **kwargs):
    """Create a quantize emulate wrapper for a keras layer.

    Args:
      layer: The keras layer to be quantized.
      quantize_config: `QuantizeConfig` to quantize layer.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
    """
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
      kwargs['name'] = self._make_layer_name(layer)

    super(QuantizeWrapper, self).__init__(layer, **kwargs)
    self.quantize_config = quantize_config
    self._mode = mode

    self._track_trackable(layer, name='layer')

  @staticmethod
  def _make_layer_name(layer):
    return '{}_{}'.format('quant', layer.name)

  @staticmethod
  def _weight_name(name):
    """Extracts the weight name from the full TensorFlow variable name.

    For example, returns 'kernel' for 'dense_2/kernel:0'.

    Args:
      name: TensorFlow variable name.

    Returns:
      Extracted weight name.
    """
    return name.split(':')[0].split('/')[-1]

  def build(self, input_shape):
    super(QuantizeWrapper, self).build(input_shape)

    self.optimizer_step = self.add_weight(
        'optimizer_step',
        initializer=tf.keras.initializers.Constant(-1),
        dtype=tf.dtypes.int32,
        trainable=False)

    self._weight_vars = []
    for weight, quantizer in \
        self.quantize_config.get_weights_and_quantizers(self.layer):
      # In case of layers without bias
      if weight is None:
        continue

      quantizer_vars = quantizer.build(weight.shape,
                                       self._weight_name(weight.name), self)

      self._weight_vars.append((weight, quantizer, quantizer_vars))
      # Needed to ensure unquantized weights get trained as part of the wrapper.
      self._trainable_weights.append(weight)

    self._quantize_activations = []
    for activation, quantizer in \
        self.quantize_config.get_activations_and_quantizers(self.layer):
      quantize_activation = vitis_quantize_aware_activation.QuantizeAwareActivation(
          activation, quantizer, self.mode, self.optimizer_step, self)
      self._quantize_activations.append(quantize_activation)

    self._output_quantizer_vars = []
    for output_id, quantizer in self.quantize_config.get_output_quantizers(
        self.layer):
      quantizer_vars = quantizer.build(
          self.layer.compute_output_shape(input_shape),
          'output_' + str(output_id), self)
      self._output_quantizer_vars.append((output_id, quantizer, quantizer_vars))

  def compute_output_shape(self, input_shape):
    return self.layer.compute_output_shape(self.layer.input_shape)

  def get_quantize_info(self):

    quantize_info = {}
    for weight, quantizer, quantizer_vars in self._weight_vars:
      w_name = weight.name.lstrip('quant_')
      quantize_info[w_name] = {
          'type': 'weight',
          'info': quantizer.get_quantize_info()
      }

    for activation in self._quantize_activations:
      name = activation._name(activation.activation)
      quantize_info[name] = activation.get_quantize_info()

    for output_id, quantizer, quantizer_vars in self._output_quantizer_vars:
      output_str = 'output_' + str(output_id)
      quantize_info[output_str] = {
          'type': 'output',
          'info': quantizer.get_quantize_info()
      }

    return quantize_info

  def set_quantize_info(self, new_quantize_info):
    for weight, quantizer, quantizer_vars in self._weight_vars:
      w_name = weight.name.lstrip('quant_')
      if w_name in new_quantize_info:
        quantizer.set_quantize_info(new_quantize_info[w_name]['info'])
      else:
        logger.warning(
            'Weight quantize_info of {} not found in new_quantize_info {}'
            .format(w_name, new_quantize_info))

    for activation in self._quantize_activations:
      name = activation._name(activation.activation)
      if name in new_quantize_info:
        activation.set_quantize_info(new_quantize_info[name])
      else:
        logger.warning(
            'Activation quantize_info of {} not found in new_quantize_info {}'
            .format(name, new_quantize_info))

    for output_id, quantizer, quantizer_vars in self._output_quantizer_vars:
      output_str = 'output_' + str(output_id)
      if output_str in new_quantize_info:
        quantizer.set_quantize_info(new_quantize_info[output_str]['info'])
      else:
        logger.warning(
            'Output quantize_info of {} not found in new_quantize_info {}'
            .format(name, new_quantize_info))

  def _make_quantizer_fn(self, quantizer, x, training, mode, quantizer_vars):
    """Use currying to return True/False specialized fns to the cond."""

    def quantizer_fn():
      return quantizer(x, training, mode, weights=quantizer_vars)

    return quantizer_fn

  def call(self, inputs, training=None):
    if training is None:
      training = tf.keras.backend.learning_phase()

    # Quantize all weights, and replace them in the underlying layer.
    quantized_weights = []
    for unquantized_weight, quantizer, quantizer_vars in self._weight_vars:
      quantized_weight = tf_utils.smart_cond(
          training,
          self._make_quantizer_fn(quantizer, unquantized_weight, True,
                                  self.mode, quantizer_vars),
          self._make_quantizer_fn(quantizer, unquantized_weight, False,
                                  self.mode, quantizer_vars))
      quantized_weights.append(quantized_weight)

    self.quantize_config.set_quantize_weights(self.layer, quantized_weights)

    # Replace all activations with `QuantizeAwareActivation`s which can
    # quantize activation tensors during graph construction.

    for quantize_activation in self._quantize_activations:
      quantize_activation.training = training

    self.quantize_config.set_quantize_activations(self.layer,
                                                  self._quantize_activations)

    args = tf_inspect.getfullargspec(self.layer.call).args
    if 'training' in args:
      outputs = self.layer.call(inputs, training=training)
    else:
      outputs = self.layer.call(inputs)

    if not self._output_quantizer_vars:
      return outputs

    # Handle layers with multiple outputs
    if isinstance(outputs, list) or isinstance(outputs, tuple):
      quantized_outputs = outputs
      for output_id, output_quantizer, output_quantizer_vars in self._output_quantizer_vars:
        quantized_outputs[output_id] = tf_utils.smart_cond(
            training,
            self._make_quantizer_fn(output_quantizer, outputs[output_id], True,
                                    self.mode, output_quantizer_vars),
            self._make_quantizer_fn(output_quantizer, outputs,
                                    False [output_id], self.mode,
                                    output_quantizer_vars))
      return quantized_outputs

    #(TODO) Handle layers with multiple outputs
    output_id, output_quantizer, output_quantizer_vars = self._output_quantizer_vars[
        0]
    return tf_utils.smart_cond(
        training,
        self._make_quantizer_fn(output_quantizer, outputs, True, self.mode,
                                output_quantizer_vars),
        self._make_quantizer_fn(output_quantizer, outputs, False, self.mode,
                                output_quantizer_vars))

  def get_config(self):
    base_config = super(QuantizeWrapper, self).get_config()
    config = {
        'quantize_config': serialize_keras_object(self.quantize_config),
        'mode': self.mode
    }
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    config = config.copy()

    layer = tf.keras.layers.deserialize(config.pop('layer'))

    # QuantizeWrapper may be constructed with any QuantizeConfig and the
    # wrapper itself cannot know all the possible config classes.
    # The deserialization code should ensure the QuantizeConfig is in keras
    # serialization scope.
    quantize_config = deserialize_keras_object(
        config.pop('quantize_config'),
        module_objects=globals(),
        custom_objects=None)

    mode = config.pop('mode')

    return cls(
        layer=layer, quantize_config=quantize_config, mode=mode, **config)

  @property
  def mode(self):
    return self._mode

  @mode.setter
  def mode(self, value):
    for quantize_activation in self._quantize_activations:
      quantize_activation.mode = value
    self._mode = value

  @property
  def trainable_weights(self):
    return self.layer.trainable_weights + self._trainable_weights

  @property
  def non_trainable_weights(self):
    return self.layer.non_trainable_weights + self._non_trainable_weights

  @property
  def updates(self):
    return self.layer.updates + self._updates

  @property
  def losses(self):
    return self.layer.losses + self._losses
