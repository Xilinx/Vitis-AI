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
"""Vitis Quantize Configurations."""

from tensorflow.python.keras.utils.generic_utils import register_keras_serializable
from tensorflow_model_optimization.python.core.quantization.keras.vitis.base import quantize_config
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantizers
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

QuantizeConfig = quantize_config.QuantizeConfig
logger = common_utils.VAILogger


def _make_quantizer(quantizer_type_name, quantizer_params):
  try:
    quantizer_cls = getattr(vitis_quantizers, quantizer_type_name)
    quantizer = quantizer_cls(**quantizer_params)
  except Exception as e:
    logger.error(
        'Fail to make quantizer `{}` with params `{}`, error: {}'.format(
            quantizer_type_name, quantizer_params, e))
  return quantizer


@register_keras_serializable(package='Vitis', name='VitisQuantizeConfig')
class VitisQuantizeConfig(QuantizeConfig):
  """QuantizeConfig for non recurrent Keras layers."""

  def __init__(self,
               quantizable_weights=[],
               weight_quantizers=[],
               quantizable_activations=[],
               activation_quantizers=[],
               quantizable_outputs=[],
               output_quantizers=[]):

    def _check_equal_len(quantizables, quantizers, name):
      if len(quantizables) != len(quantizers):
        logger.error('Length of quantizable_{}s and {}_quantizers '
                     'should be the same, but {} and {} are given.'.format(
                         name, name, len(quantizables), len(quantizers)))

    _check_equal_len(quantizable_weights, weight_quantizers, 'weight')
    _check_equal_len(quantizable_activations, activation_quantizers,
                     'activation')
    _check_equal_len(quantizable_outputs, output_quantizers, 'output')

    self.quantizable_weights = quantizable_weights
    self.weight_quantizers = weight_quantizers
    self._weight_quantizers = []
    for quantizer in weight_quantizers:
      self._weight_quantizers.append(
          _make_quantizer(quantizer['quantizer_type'],
                          quantizer['quantizer_params']))

    self.quantizable_activations = quantizable_activations
    self.activation_quantizers = activation_quantizers
    self._activation_quantizers = []
    for quantizer in activation_quantizers:
      self._activation_quantizers.append(
          _make_quantizer(quantizer['quantizer_type'],
                          quantizer['quantizer_params']))

    self.quantizable_outputs = quantizable_outputs
    self.output_quantizers = output_quantizers
    self._output_quantizers = []
    for quantizer in output_quantizers:
      self._output_quantizers.append(
          _make_quantizer(quantizer['quantizer_type'],
                          quantizer['quantizer_params']))

  def get_quantizable_weights(self):
    return self.quantizable_weights

  def get_weight_quantizers(self):
    return self._weight_quantizers

  def get_quantizable_activations(self):
    return self.quantizable_activations

  def get_activation_quantizers(self):
    return self._activation_quantizers

  def get_quantizable_outputs(self):
    return self.quantizable_outputs

  def get_output_quantizers(self):
    return self._output_quantizers

  def get_weights_and_quantizers(self, layer):
    return [(getattr(layer, weight), quantizer) for weight, quantizer in zip(
        self.quantizable_weights, self._weight_quantizers)]

  def get_activations_and_quantizers(self, layer):
    return [(getattr(layer, activation), quantizer) for activation, quantizer in
            zip(self.quantizable_activations, self._activation_quantizers)]

  def set_quantize_weights(self, layer, quantize_weights):
    for weight, quantize_weight in zip(self.quantizable_weights,
                                       quantize_weights):
      current_weight = getattr(layer, weight)
      if current_weight.shape != quantize_weight.shape:
        logger.error('Existing layer weight shape {} is incompatible with '
                     'provided quantize weight shape {}'.format(
                         current_weight.shape, quantize_weight.shape))

      setattr(layer, weight, quantize_weight)

  def set_quantize_activations(self, layer, quantize_activations):
    if len(self.quantizable_activations) != len(quantize_activations):
      logger.error('`set_quantize_activations` called on layer {} with {} '
                   'activation parameters, but layer expects {} values.'.format(
                       layer.name, len(quantize_activations),
                       len(self.quantizable_activations)))

    for activation, quantize_activation in \
        zip(self.quantizable_activations, quantize_activations):
      setattr(layer, activation, quantize_activation)

  def get_output_quantizers(self, layer):
    return [(output_id, quantizer) for output_id, quantizer in zip(
        self.quantizable_outputs, self._output_quantizers)]

  @classmethod
  def from_config(cls, config):
    """Instantiates a `VitisQuantizeConfig` from its config.

    Args:
        config: Output of `get_config()`.

    Returns:
        A `VitisQuantizeConfig` instance.
    """
    return cls(**config)

  def get_config(self):
    return {
        'quantizable_weights': self.quantizable_weights,
        'weight_quantizers': self.weight_quantizers,
        'quantizable_activations': self.quantizable_activations,
        'activation_quantizers': self.activation_quantizers,
        'quantizable_outputs': self.quantizable_outputs,
        'output_quantizers': self.output_quantizers
    }

  def __eq__(self, other):
    if not isinstance(other, VitisQuantizeConfig):
      return False

    return (self.quantizable_weights == other.quantizable_weights and
            self.weight_quantizers == other.weight_quantizers and
            self.quantizable_activations == self.quantizable_activations and
            self.activation_quantizers == other.activation_quantizers and
            self.quantizable_outputs == other.quantizable_outputs and
            self.output_quantizers == other.output_quantizers)

  def __ne__(self, other):
    return not self.__eq__(other)


@register_keras_serializable(package='Vitis', name='NoQuantizeConfig')
class NoQuantizeConfig(QuantizeConfig):
  """QuantizeConfig which does not quantize any part of the layer.
  
  This is used in vitis_transforms.py to ensure no quantization inside
  some patterns.
  """

  def get_weights_and_quantizers(self, layer):
    return []

  def get_activations_and_quantizers(self, layer):
    return []

  def set_quantize_weights(self, layer, quantize_weights):
    pass

  def set_quantize_activations(self, layer, quantize_activations):
    pass

  def get_output_quantizers(self, layer):
    return []

  def get_config(self):
    return {}


def _types_dict():
  return {
      'VitisQuantizeConfig': VitisQuantizeConfig,
      'NoQuantizeConfig': NoQuantizeConfig,
      'Vitis8BitQuantizeConfig': VitisQuantizeConfig,
  }
