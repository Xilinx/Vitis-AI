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
"""Qauntize Keras layers with wrapper."""

import os
import collections
import copy
import datetime
import random
import string

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_custom_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_configs
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantizers
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_quantize
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_pooling
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_conv_bn
from tensorflow_model_optimization.python.core.quantization.keras.vitis.graph_transformations import transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.optimizations import vitis_optimize_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy import vitis_quantize_strategy_factory

from .quant_subclass_layers import QuantIdentity


class KerasLayersWrapper(object):
  """Quantize keras layers with wrapper."""

  def __init__(self, quantize_registry, mode):
    """Create a quantize wrapper for a keras layer."""

    self.quantize_registry = quantize_registry
    self.mode = mode

  def remove(self, layer):
    """Remove a keras layer by replacing it with Identity layer."""

    if not isinstance(layer, tf.keras.layers.Layer):
      return None

    if layer.__class__.__name__ == "Dropout":
      return QuantIdentity(
          application='none', name=layer.name, dtype=layer.dtype)
    else:
      return None

  def apply(self, layer):
    """Apply a quantize wrapper for a keras layer."""

    if self.quantize_registry is None or \
       self.mode not in ['QCB', 'QAT', 'ANALYSE', 'QCBEV']:
      return None

    if not isinstance(layer, tf.keras.layers.Layer):
      return None

    # Replace special layers
    if layer.__class__.__name__ == "QuantIdentity":
      quantize_config = self.quantize_registry.get_input_quantize_config()

      if layer.application == 'input':
        input_quantizer = vitis_quantize_configs._make_quantizer(
            quantize_config['input_quantizer']['quantizer_type'],
            quantize_config['input_quantizer']['quantizer_params'])
        return vitis_quantize.VitisQuantize(
            input_quantizer,
            self.mode,
            name="quant_" + layer.name,
            dtype=layer.dtype)

      elif layer.application == 'output':
        # TODO: add configuration for output_quantizer
        quantize_params = quantize_config['input_quantizer'][
            'quantizer_params'].copy()
        if quantize_config['input_quantizer']['quantizer_type'] != 'TQTQuantizer':
          quantize_params['method'] = 1  # Note TQTQuantizer does not support this

        output_quantizer = vitis_quantize_configs._make_quantizer(
            quantize_config['input_quantizer']['quantizer_type'],
            quantize_params)
        return vitis_quantize.VitisQuantize(
            output_quantizer,
            self.mode,
            name="quant_" + layer.name,
            dtype=layer.dtype)

    # Wrap standard layers
    elif self.quantize_registry.supports(layer):
      quantize_config = self.quantize_registry.get_quantize_config(layer)
      return vitis_quantize_wrapper.QuantizeWrapper(
          layer, quantize_config, self.mode, dtype=layer.dtype)

    # Unsupported layers
    return None
