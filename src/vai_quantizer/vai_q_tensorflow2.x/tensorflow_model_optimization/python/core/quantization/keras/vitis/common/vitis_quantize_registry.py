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
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_layer_limits
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
    self._layer_limits_map = dict()
    self._layer_quantize_map = dict()
    if 'layer_quantize_config' in configs:
      for config in configs['layer_quantize_config']:
        self._update_layer_quantize_map(config)

  def _parse_layer_type(self, layer_type_str):
    try:
      module = layer_type_str.rsplit('.', 1)
      if len(module) == 1:
        layer_type = eval(module[0])
      else:
        module_name, layer_name = module
        layer_module = importlib.import_module(module_name)
        layer_type = getattr(layer_module, layer_name)
    except Exception as e:
        logger.error('[Quantizer_TF2_Unsupported_Layer][Unsupported layer type]'
                     'Fail to parse layer type `{}`, error: {}'.format(
          layer_type_str, e))
    return layer_type

  def _parse_layer_quantize_config(self, config):
    config = copy.deepcopy(config)
    if 'layer_type' in config:
      config.pop('layer_type')
    return vitis_quantize_configs.VitisQuantizeConfig.from_config(config)

  def _update_layer_quantize_map(self, new_config):
    layer_type = self._parse_layer_type(new_config['layer_type'])
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

  def _apply_user_quantize_config(self,
                                  layer_type,
                                  layer_config,
                                  layer_name=None,
                                  layer_relulike=False):
    """Apply user quantize config to layer quantize config."""
    layer_config = copy.copy(layer_config)
    user_config = self.get_user_quantize_config()

    def _override_quantizer_params(quantizer,
                                   user_config,
                                   user_key,
                                   local_key,
                                   name=''):
      user_value = user_config.get(user_key, None)
      local_value = quantizer['quantizer_params'].get(local_key, None)
      if None not in [
          user_value, local_value
      ] and quantizer['quantizer_params'][local_key] != user_value:
        quantizer['quantizer_params'][local_key] = user_value
        logger.debug('Override default {} of {}({})\'s {}: {} -> {}'.format(
            user_key, layer_name, layer_type, name, local_value, user_value))

    def _override_activation_unsigned(quantizer,
                                      user_config,
                                      user_key,
                                      local_key,
                                      name='',
                                      relu_like=False):
      ''' Non relu-like layer cannot set unsigned when symmetry quantization'''
      symmetry_value = quantizer['quantizer_params'].get('symmetry', None)
      unsigned_value = user_config.get('activation_unsigned', None)
      if None not in [
          symmetry_value, unsigned_value
      ] and symmetry_value and unsigned_value and relu_like == False:
        logger.debug(
            'Override ignored {} of {}({})\'s {}: symmetry {}, unsigned {}'
            .format(user_key, layer_name, layer_type, name, symmetry_value,
                    unsigned_value))
      else:
        _override_quantizer_params(quantizer, user_config, user_key, local_key,
                                   name)

    quantizer = layer_config.get('input_quantizer')
    if quantizer:
      _override_quantizer_params(quantizer, user_config, 'input_bit',
                                 'bit_width', 'input')
      _override_quantizer_params(quantizer, user_config, 'input_method',
                                 'method', 'input')
      _override_quantizer_params(quantizer, user_config,
                                 'input_method_percentile', 'method_percentile',
                                 'input')
      _override_quantizer_params(quantizer, user_config, 'input_per_channel',
                                 'per_channel', 'input')
      _override_quantizer_params(quantizer, user_config, 'input_symmetry',
                                 'symmetry', 'input')
      _override_quantizer_params(quantizer, user_config, 'input_round_mode',
                                 'round_mode', 'input')
      _override_quantizer_params(quantizer, user_config, 'input_unsigned',
                                 'unsigned', 'input')
      _override_quantizer_params(quantizer, user_config, 'use_framework_quant',
                                 'use_framework_quant', 'input')

    for quantizer in layer_config.get('weight_quantizers', []):
      _override_quantizer_params(quantizer, user_config, 'weight_bit',
                                 'bit_width', 'weight')
      _override_quantizer_params(quantizer, user_config, 'weight_method',
                                 'method', 'weight')
      _override_quantizer_params(quantizer, user_config, 'weight_per_channel',
                                 'per_channel', 'weight')
      _override_quantizer_params(quantizer, user_config, 'weight_symmetry',
                                 'symmetry', 'weight')
      _override_quantizer_params(quantizer, user_config, 'weight_round_mode',
                                 'round_mode', 'weight')
      _override_quantizer_params(quantizer, user_config, 'weight_unsigned',
                                 'unsigned', 'weight')
      _override_quantizer_params(quantizer, user_config, 'use_framework_quant',
                                 'use_framework_quant', 'weight')

    for quantizer in layer_config.get('bias_quantizers', []):
      _override_quantizer_params(quantizer, user_config, 'bias_bit',
                                 'bit_width', 'bias')
      _override_quantizer_params(quantizer, user_config, 'bias_method',
                                 'method', 'bias')
      _override_quantizer_params(quantizer, user_config, 'bias_per_channel',
                                 'per_channel', 'bias')
      _override_quantizer_params(quantizer, user_config, 'bias_symmetry',
                                 'symmetry', 'bias')
      _override_quantizer_params(quantizer, user_config, 'bias_round_mode',
                                 'round_mode', 'bias')
      _override_quantizer_params(quantizer, user_config, 'bias_unsigned',
                                 'unsigned', 'bias')
      _override_quantizer_params(quantizer, user_config, 'use_framework_quant',
                                 'use_framework_quant', 'bias')

    for quantizer in layer_config.get('activation_quantizers', []):
      _override_quantizer_params(quantizer, user_config, 'activation_bit',
                                 'bit_width', 'activation')
      _override_quantizer_params(quantizer, user_config, 'activation_method',
                                 'method', 'activation')
      _override_quantizer_params(quantizer, user_config,
                                 'activation_method_percentile',
                                 'method_percentile', 'activation')
      _override_quantizer_params(quantizer, user_config,
                                 'activation_per_channel', 'per_channel',
                                 'activation')
      _override_quantizer_params(quantizer, user_config, 'activation_symmetry',
                                 'symmetry', 'activation')
      _override_quantizer_params(quantizer, user_config,
                                 'activation_round_mode', 'round_mode',
                                 'activation')
      _override_activation_unsigned(quantizer, user_config,
                                    'activation_unsigned', 'unsigned',
                                    'activation', layer_relulike)
      _override_quantizer_params(quantizer, user_config, 'use_framework_quant',
                                 'use_framework_quant', 'activation')

    for quantizer in layer_config.get('output_quantizers', []):
      _override_quantizer_params(quantizer, user_config, 'activation_bit',
                                 'bit_width', 'output')
      _override_quantizer_params(quantizer, user_config, 'activation_method',
                                 'method', 'output')
      _override_quantizer_params(quantizer, user_config,
                                 'activation_method_percentile',
                                 'method_percentile', 'output')
      _override_quantizer_params(quantizer, user_config,
                                 'activation_per_channel', 'per_channel',
                                 'output')
      _override_quantizer_params(quantizer, user_config, 'activation_symmetry',
                                 'symmetry', 'output')
      _override_quantizer_params(quantizer, user_config,
                                 'activation_round_mode', 'round_mode',
                                 'output')
      _override_activation_unsigned(quantizer, user_config,
                                    'activation_unsigned', 'unsigned', 'output',
                                    layer_relulike)
      _override_quantizer_params(quantizer, user_config, 'use_framework_quant',
                                 'use_framework_quant', 'output')

    return layer_config

  def _get_quantize_config(self,
                           layer_type,
                           layer_name=None,
                           layer_relulike=False):
    quantize_config = copy.deepcopy(self._layer_quantize_map.get(layer_type))
    config = quantize_config.get_config()
    config = self._apply_user_quantize_config(layer_type, config, layer_name,
                                              layer_relulike)
    return self._parse_layer_quantize_config(config)

  def _is_supported_layer(self, layer_type):
    return layer_type in self._layer_quantize_map

  def _relu_like_layer(self, layer):
    layer_relulike = False
    if isinstance(layer, tf.keras.layers.ReLU):
      layer_relulike = True
    else:
      layer_config = layer.get_config()
      if 'activation' in layer_config and layer_config['activation'] in [
          'relu', 'relu6'
      ]:
        layer_relulike = True
    return layer_relulike

  # Interface functions.

  def get_configs(self):
    """Get the configurations."""
    return self._configs

  def print_configs(self, verbose=0):
    """Print the configurations."""
    for k, v in self._configs.items():
      if k == 'user_quantize_config':
        if verbose > 0:
          logger.info('user_quantize_config:')
        for name, value in v.items():
          if verbose > 0:
            logger.info('- {}: {}'.format(name, value))
          else:
            logger.debug('- {}: {}'.format(name, value))

      if k in ['input_quantize_config', 'layer_quantize_config']:
        logger.debug('- {}: {}'.format(k, v))

  def is_valid_config(self, config):
    """Check if the config is valid."""
    return config[0] in self.get_configs(
    ) or config[0] in self.get_user_quantize_config()

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
          'custom_layer_type', 'input_quantize_config', 'user_quantize_config'
      ]:
        self._configs.update({config[0]: config[1]})
        logger.info('Update {}: {}'.format(config[0], config[1]))

      elif config[0] in self.get_user_quantize_config():
        self.get_user_quantize_config().update({config[0]: config[1]})
        logger.info('Update {}: {}'.format(config[0], config[1]))

      elif config[0] == 'layer_quantize_config':
        for config in config[1]:
          self._update_layer_quantize_config(config)
      else:
        logger.error('Invalid config {}.'.format(config))

  def get_user_quantize_config(self):
    """Get user quantize configurations."""
    return self._configs['user_quantize_config']

  def get_symmetry(self):
    """Get symmetry configuration."""
    return self._configs['symmetry']

  def get_input_quantize_config(self):
    """Get input quantize config."""
    config = self._configs['input_quantize_config']
    config = self._apply_user_quantize_config('InputLayer', config, None)
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

  def get_quantize_config(self, layer):
    """Returns the quantization config for the given layer.

    Args:
      layer: input layer to return quantize config for.

    Returns:
      Returns the QuantizeConfig for the given layer.
    """
    if not self.supports(layer):
      logger.error(
          '[Quantizer_TF2_Unsupported_Layer][Unsupported layer type]'
          '`get_quantize_config()` called on an unsupported layer {}. Check '
          'if layer is supported by calling `supports()`. Alternatively, you '
          'can use `QuantizeConfig` to specify a behavior for your layer.'
          .format(layer.__class__))

    if self._is_supported_layer(layer.__class__):
      return self._get_quantize_config(layer.__class__, layer.name,
                                       self._relu_like_layer(layer))

    # Should never come here.
    logger.error('[Quantizer_TF2_Unsupported_Layer][Unsupported layer type]''Invalid Layer type {}'.format(layer.__class__))

  def _get_layer_limits(self, layer_type):
    """Return the layer limits for the given layer_type."""
    return self._layer_limits_map.get(layer_type)

  def get_layer_limits(self, layer):
    """Returns the layer limits for the given layer.

    Args:
      layer: input layer to return layer limits for.

    Returns:
      Returns the layer_limits for the given layer.
    """
    if layer.__class__ not in self._layer_limits_map:
      return None
    return self._get_layer_limits(layer.__class__)

  def update_layer_limits(self, layer_limits_map):
    """Update the layer_limits_map."""
    for layer_type, layer_limit in layer_limits_map.items():
      _layer_type = self._parse_layer_type(layer_type)
      if _layer_type not in self._layer_limits_map:
        self._layer_limits_map[_layer_type] = layer_limit
      else:
        self._layer_limits_map[_layer_type].merget(layer_limit)
