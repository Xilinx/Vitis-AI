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
"""Vitis trained quantization threshold post-quantization refine transforms."""

import tensorflow as tf
import numpy as np
import collections

from tensorflow_model_optimization.python.core.quantization.keras.vitis.graph_transformations import transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantizers
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

serialize_keras_object = tf.keras.utils.serialize_keras_object
deserialize_keras_object = tf.keras.utils.deserialize_keras_object
LayerPattern = transforms.LayerPattern
logger = common_utils.VAILogger


def _make_quantizer(quantizer_type_name, quantizer_params):
  try:
    quantizer_cls = getattr(vitis_quantizers, quantizer_type_name)
    quantizer = quantizer_cls(**quantizer_params)
  except Exception as e:
    logger.error(
        '[Quantizer_TF2_Unsupported_Layer][Unsupported layer type] '
        'Fail to make quantizer `{}` with params `{}`, error: {}'.format(
            quantizer_type_name, quantizer_params, e))
  return quantizer


class ConvertTQTToPof2SQuantizeStrategy(transforms.Transform):
  """Convert quantize strategy of quantized layers from tqt to pof2s."""

  def __init__(self):
    super(ConvertTQTToPof2SQuantizeStrategy, self).__init__()

  def pattern(self):
    return LayerPattern('Vitis>VitisQuantize|Vitis>QuantizeWrapper', {}, [])

  def replacement(self, match_layer):
    layer_type = match_layer.layer['class_name']

    if layer_type == 'Vitis>VitisQuantize':
      quantizer = match_layer.layer['config']['quantizer']
      if quantizer['class_name'] == 'Vitis>TQTQuantizer':
        tqt_quantizer = deserialize_keras_object(quantizer)
        pof2s_quantizer = tqt_quantizer.convert_to_pof2s_quantizer()
        quantizer.update(serialize_keras_object(pof2s_quantizer))
    elif layer_type == 'Vitis>QuantizeWrapper':
      quantize_config = match_layer.layer['config']['quantize_config']
      if not quantize_config['class_name'] == 'Vitis>NoQuantizeConfig':
        config = quantize_config['config']
        quantizers = config['weight_quantizers'] + config[
            'bias_quantizers'] + config['activation_quantizers'] + config[
                'output_quantizers']
        for quantizer in quantizers:
          if quantizer['quantizer_type'] == 'TQTQuantizer':
            tqt_quantizer = _make_quantizer(quantizer['quantizer_type'],
                                            quantizer['quantizer_params'])
            pof2s_quantizer = tqt_quantizer.convert_to_pof2s_quantizer()
            quantizer.update({
                'quantizer_type': 'Pof2SQuantizer',
                'quantizer_params': pof2s_quantizer.get_config()
            })

    match_layer.weights = self._convert_weights(match_layer.weights)
    return match_layer

  def _convert_weights(self, weights):
    """Helper function to convert weights from TQTQuantizer to Pof2SQuantizer."""
    new_weights = collections.OrderedDict()
    for k, v in weights.items():
      if k.endswith('log_th:0'):
        name = k.replace('log_th', 'pos')
        log_th = v
        pos = 7 - np.ceil(log_th)
        new_weights[name] = pos
      else:
        new_weights[k] = v
    return new_weights

  def custom_objects(self):
    return {}
