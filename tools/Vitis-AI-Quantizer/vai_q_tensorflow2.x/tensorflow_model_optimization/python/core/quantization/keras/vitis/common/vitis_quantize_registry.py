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
import importlib
import copy

from tensorflow_model_optimization.python.core.quantization.keras.vitis.base import quantize_config
from tensorflow_model_optimization.python.core.quantization.keras.vitis.base import quantize_registry
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_configs
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

# Register the vitis built-in layers
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_quantize
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_pooling

QuantizeConfig = quantize_config.QuantizeConfig
QuantizeRegistry = quantize_registry.QuantizeRegistry
logger = common_utils.VAILogger


class VitisQuantizeRegistry(QuantizeRegistry):
  """QuantizationRegistry for built-in Keras and Vitis layers."""

  def __init__(self, configs):
    """Init."""
    self._configs = configs
    self._layer_quantize_map = dict()
    if 'layer_quantize_config' in configs:
      for config in configs['layer_quantize_config']:
        self._update_layer_quantize_map(config)

  def _parse_layer_type(self, config):
    try:
      module = config['layer_type'].rsplit('.', 1)
      if len(module) == 1:
        layer_type = eval(module[0])
      else:
        module_name, layer_name = module
        layer_module = importlib.import_module(module_name)
        layer_type = getattr(layer_module, layer_name)
    except Exception as e:
      logger.error('Fail to parse layer type `{}`, error: {}'.format(
          config['layer_type'], e))
    return layer_type

  def _parse_layer_quantize_config(self, config):
    config = copy.deepcopy(config)
    if 'layer_type' in config:
      config.pop('layer_type')
    return vitis_quantize_configs.VitisQuantizeConfig.from_config(config)

  def _update_layer_quantize_map(self, new_config):
    layer_type = self._parse_layer_type(new_config)
    layer_quantize_config = self._parse_layer_quantize_config(new_config)
    self._layer_quantize_map[layer_type] = layer_quantize_config

  def _update_layer_quantize_config(self, new_config):
    found = False
    for i, config in enumerate(self._configs['layer_quantize_config']):
      if new_config['layer_type'] == config['layer_type']:
        logger.info('Update layer_quantize_config {}'.format(new_config))
        self._configs['layer_quantize_config'][i] = new_config
        found = True

    if not found:
      logger.info('Add new layer_quantize_config {}'.format(new_config))
      self._configs['layer_quantize_config'].append(new_config)

    self._update_layer_quantize_map(new_config)

  def _is_supported_layer(self, layer_type):
    return layer_type in self._layer_quantize_map

  # Interface functions.

  def get_configs(self):
    """Get the configurations."""
    return self._configs

  def print_configs(self):
    """Print the configurations."""
    for k, v in self._configs.items():
      if not k in ['input_quantize_config', 'layer_quantize_config'] and v > 0:
        logger.debug('- {}: {}'.format(k, v))

  def is_valid_config(self, config):
    """Check if the config is valid."""
    return config[0] in self.get_configs()

  def update(self, configs):
    """Update quantize registry configurations."""

    if not isinstance(configs, dict):
      if isinstance(configs, tuple) and len(configs) == 2:
        configs = {configs[0]: configs[1]}
      else:
        logger.error('Invalid format of configs: {}'.format(configs))

    for config in configs.items():
      if not self.is_valid_config(config):
        logger.error('Invalid config {} for {}.'.format(config, self.__class__))

      if config[0] in [
          'input_bit', 'weight_bit', 'activation_bit', 'input_quantize_config'
      ]:
        self._configs.update({config[0]: config[1]})
        logger.info('Update {}: {}'.format(config[0], config[1]))

      elif config[0] == 'layer_quantize_config':
        for config in config[1]:
          self._update_layer_quantize_config(config)
      else:
        logger.error('Invalid config {}.'.format(config))

  def get_input_bit(self):
    """Get input bit width."""
    if self._configs['input_bit'] > 0:
      return self._configs['input_bit']
    else:
      return None

  def get_weight_bit(self):
    """Get weight bit width."""
    if self._configs['weight_bit'] > 0:
      return self._configs['weight_bit']
    else:
      return None

  def get_activation_bit(self):
    """Get activation bit width."""
    if self._configs['activation_bit'] > 0:
      return self._configs['activation_bit']
    else:
      return None

  def get_input_quantize_config(self):
    """Get input quantize config."""
    config = self._configs['input_quantize_config']
    input_bit = self.get_input_bit()
    if input_bit:
      config['input_quantizer']['quantizer_params']['bit_width'] = input_bit
      logger.debug('Override default bit_width: input -> {}'.format(input_bit))
    return config

  def supports(self, layer):
    """Returns whether the registry supports this layer type.

    Args:
      layer: The layer to check for support.

    Returns:
      True/False whether the layer type is supported.

    """
    if self._is_supported_layer(layer.__class__):
      return True

    return False

  def _get_quantize_config(self, layer_type):
    quantize_config = copy.deepcopy(self._layer_quantize_map.get(layer_type))

    # Override bit width with global settings
    weight_bit = self.get_weight_bit()
    activation_bit = self.get_activation_bit()
    if weight_bit or activation_bit:
      config = quantize_config.get_config()
      if weight_bit:
        for quantizer in config['weight_quantizers']:
          old_weight_bit = quantizer['quantizer_params']['bit_width']
          quantizer['quantizer_params']['bit_width'] = weight_bit
          logger.debug('Override default bit_width: {}:weight {} -> {}'.format(
              layer_type, old_weight_bit, weight_bit))
      if activation_bit:
        for quantizer in config['activation_quantizers']:
          old_activation_bit = quantizer['quantizer_params']['bit_width']
          quantizer['quantizer_params']['bit_width'] = activation_bit
          logger.debug(
              'Override default bit_width: {}:activation {} -> {}'.format(
                  layer_type, old_activation_bit, activation_bit))
        for quantizer in config['output_quantizers']:
          old_output_bit = quantizer['quantizer_params']['bit_width']
          quantizer['quantizer_params']['bit_width'] = activation_bit
          logger.debug('Override default bit_width: {}:output {} -> {}'.format(
              layer_type, old_output_bit, activation_bit))
      return self._parse_layer_quantize_config(config)

    return quantize_config

  def get_quantize_config(self, layer):
    """Returns the quantization config for the given layer.

    Args:
      layer: input layer to return quantize config for.

    Returns:
      Returns the QuantizeConfig for the given layer.
    """
    if not self.supports(layer):
      logger.error(
          '`get_quantize_config()` called on an unsupported layer {}. Check '
          'if layer is supported by calling `supports()`. Alternatively, you '
          'can use `QuantizeConfig` to specify a behavior for your layer.'
          .format(layer.__class__))

    if self._is_supported_layer(layer.__class__):
      return self._get_quantize_config(layer.__class__)

    # Should never come here.
    logger.error('Invalid Layer type {}'.format(layer.__class__))
