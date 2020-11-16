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
"""Quantization registry which specifies how layers should be quantized."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import json
import importlib

from tensorflow_model_optimization.python.core.quantization.keras.vitis.base import quantize_config
from tensorflow_model_optimization.python.core.quantization.keras.vitis.base import quantize_registry
from tensorflow_model_optimization.python.core.quantization.keras.vitis import vitis_8bit_quantize_configs

QuantizeConfig = quantize_config.QuantizeConfig
layers = tf.keras.layers

dirname = os.path.dirname(__file__)
vitis_8bit_default_quantize_strategy_json = os.path.join(
    dirname, "vitis_8bit_default_quantize_strategy.json")


class QuantizeStrategy(object):
  """Quantize Strategy."""

  def __init__(self, layer_type, quantizable_weights, weight_quantizers,
               quantizable_activations, activation_quantizers,
               quantizable_outputs, output_quantizers):
    """Quantize Strategy.

    Args:
      layer_type: Type of keras layer.
      quantizable_weights: List of quantizable weight attributes of layer.
      weight_quantizers: List of quantizer for each weight.
      quantizable_activations: List of quantizable activation attributes of layer.
      activation_quantizers: List of quantizer for each activation.
      quantizable_outputs: List of quantizable outputs of layer.
      output_quantizers: List of quantizer for each output.
    """
    self.layer_type = layer_type
    self.quantizable_weights = quantizable_weights
    self.weight_quantizers = weight_quantizers
    self.quantizable_activations = quantizable_activations
    self.activation_quantizers = activation_quantizers
    self.quantizable_outputs = quantizable_outputs
    self.output_quantizers = output_quantizers


def load_json(json_file):
  with open(json_file, 'r') as f:
    try:
      data = json.loads(f.read())
    except Exception as e:
      raise ValueError(
          'Fail to load the json file `{}`, please check the format. \nError: {}'
          .format(json_file, e))
  return data


class QuantizeRegistry(quantize_registry.QuantizeRegistry):
  """QuantizationRegistry for built-in Keras classes for vitis 8-bit scheme."""

  def __init__(self):
    self._layer_quantize_map = dict()
    self.update_quantize_strategy(vitis_8bit_default_quantize_strategy_json)

  def update_quantize_strategy(self, quantize_strategy_file):
    quantize_strategies = load_json(quantize_strategy_file)
    if 'input_quantizer' in quantize_strategies:
      self._input_quantizer = quantize_strategies['input_quantizer']

    new_layer_quantize_map = dict()
    for qs in quantize_strategies['quantize_strategies']:
      try:
        layer_module = importlib.import_module(qs['layer_module'])
        layer_type = getattr(layer_module, qs['layer_type'])
      except Exception as e:
        raise ValueError(
            'Fail to get layer type `{}` from module `{}`, error: {}'.format(
                qs['layer_type'], qs['layer_module'], e))
      quantizable_weights = qs.get('quantizable_weights', [])
      weight_quantizers = qs.get('weight_quantizers', [])
      quantizable_activations = qs.get('quantizable_activations', [])
      activation_quantizers = qs.get('activation_quantizers', [])
      quantizable_outputs = qs.get('quantizable_outputs', [])
      output_quantizers = qs.get('output_quantizers', [])
      new_layer_quantize_map[layer_type] = QuantizeStrategy(
          layer_type, quantizable_weights, weight_quantizers,
          quantizable_activations, activation_quantizers, quantizable_outputs,
          output_quantizers)

    return self._layer_quantize_map.update(new_layer_quantize_map)

  def _is_supported_layer(self, layer_type):
    return layer_type in self._layer_quantize_map

  def _get_quantize_strategy(self, layer_type):
    return self._layer_quantize_map[layer_type]

  # Interface functions.
  def get_input_quantizer(self):
    return self._input_quantizer

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

    return False

  def _get_quantize_config(self, layer_type):
    qs = self._get_quantize_strategy(layer_type)

    # In case of hard coded quantize config
    if isinstance(qs, QuantizeConfig):
      return qs

    return vitis_8bit_quantize_configs.Vitis8BitQuantizeConfig(
        qs.quantizable_weights, qs.weight_quantizers,
        qs.quantizable_activations, qs.activation_quantizers,
        qs.quantizable_outputs, qs.output_quantizers)

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

    # Should never come here.
    raise ValueError('Invalid Layer type {}'.format(layer.__class__))
