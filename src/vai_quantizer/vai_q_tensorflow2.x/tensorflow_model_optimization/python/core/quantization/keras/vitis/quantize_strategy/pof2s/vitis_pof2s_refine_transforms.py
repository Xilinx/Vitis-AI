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
"""Vitis power-of-2 scale post-quantization refine transforms."""

import collections
import inspect
import copy

import tensorflow as tf
import numpy as np

from tensorflow_model_optimization.python.core.quantization.keras.vitis.graph_transformations import transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantizers
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_quantize as vitis_quantize_layer
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

serialize_keras_object = tf.keras.utils.serialize_keras_object
deserialize_keras_object = tf.keras.utils.deserialize_keras_object
LayerPattern = transforms.LayerPattern
logger = common_utils.VAILogger
keras = tf.keras


def _get_pos(layer, quantize_info, key):
  """Get the quantize pos of layer:key in quantize_info."""
  if isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper):
    q_info = quantize_info[layer.layer.name]

    # Recursive searching for layers with empty quantize info to skip some
    # special layers which are transparent to quantization, such as Reshape,
    # Flatten and ZeroPadding2D layers.
    if not q_info:
      pre_layer = layer.inbound_nodes[0].inbound_layers
      return _get_pos(pre_layer, quantize_info, key)

    for k, v in q_info.items():
      if key == 'w' and v.get('type') == 'weight' and k.endswith('kernel:0'):
        return v['info']['quant_pos_var']
      elif key == 'b' and v.get('type') == 'weight' and k.endswith('bias:0'):
        return v['info']['quant_pos_var']
      elif key == 'o':
        if v.get('type') in ['post_activation', 'pre_activation', 'output']:
          return v['info']['quant_pos_var']
  elif isinstance(layer, vitis_quantize_layer.VitisQuantize):
    if key == 'o':
      q_info = quantize_info[layer.name]
      return q_info['info']['quant_pos_var']
  else:
    return None


def _set_pos(layer, quantize_info, key, new_pos):
  """Set the quantize pos of layer:key in quantize_info to new_pos."""
  if isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper):
    q_info = quantize_info[layer.layer.name]
    for k, v in q_info.items():
      if k == 'NoQuantizeActivation':
        continue
      if key == 'w' and v.get('type') == 'weight' and k.endswith('kernel:0'):
        v['info']['quant_pos_var'] = new_pos
        return
      elif key == 'b' and v.get('type') == 'weight' and k.endswith('bias:0'):
        v['info']['quant_pos_var'] = new_pos
        return
      elif key == 'o' and v.get('type') in [
          'post_activation', 'pre_activation', 'output'
      ]:
        v['info']['quant_pos_var'] = new_pos
  elif isinstance(layer, vitis_quantize_layer.VitisQuantize):
    if key == 'o':
      q_info = quantize_info[layer.name]
      q_info['info']['quant_pos_var'] = new_pos
  return


def _get_iwob_pos(layer, quantize_info):
  """Get the input/weight/output/bias quantize pos of layer in quantize_info."""
  wp = _get_pos(layer, quantize_info, 'w')
  bp = _get_pos(layer, quantize_info, 'b')

  if hasattr(layer.layer, 'activation') and isinstance(
      layer.layer.activation.activation,
      vitis_quantize_aware_activation.NoQuantizeActivation):

    post_layer = layer.outbound_nodes[0].outbound_layer
    op = _get_pos(post_layer, quantize_info, 'o')
  else:
    op = _get_pos(layer, quantize_info, 'o')

  pre_layer = layer.inbound_nodes[0].inbound_layers
  ip = _get_pos(pre_layer, quantize_info, 'o')
  return ip, wp, op, bp


def _is_valid(ip, wp, op, bp=0):
  """Check if the input/weight/output/bias quantize pos is valid."""
  if None in [ip, op]:
    return False
  if (isinstance(wp, np.ndarray) and None in wp) or wp is None:
    return False
  if (isinstance(bp, np.ndarray) and None in bp) or bp is None:
    return False
  return True


def _adjust_shift_cut(model, quantize_info):
  """Adjust the shift cut of layer.

  shift_cut = wp + ip - op

  DPU compiler constraints of shift_cut:
    1. 0 <= shift_cut <= 16
  """
  adjusted_quantize_info = copy.deepcopy(quantize_info)

  for i in range(1, len(model.layers)):
    layer = model.layers[i]
    if not isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper):
      continue

    # Only adjust shift_cut and shift_bias for Conv-like layers
    if not (isinstance(layer.layer, keras.layers.Conv2D) or
            isinstance(layer.layer, keras.layers.DepthwiseConv2D) or
            isinstance(layer.layer, keras.layers.Conv2DTranspose) or
            isinstance(layer.layer, keras.layers.Dense)):
      continue

    ip, wp, op, bp = _get_iwob_pos(layer, adjusted_quantize_info)

    if isinstance(wp, np.ndarray):
      logger.debug('Not support adjust shift cut for per_channel quantization.')
      return adjusted_quantize_info

    if not _is_valid(ip, wp, op):
      logger.debug('Skip shift cut adjustment for layer {}, '
                   'its quantize pos is [i={}, w={}, b={}, o={}]'.format(
                       layer.name, ip, wp, bp, op))
      return adjusted_quantize_info

    min_sc = 0
    max_sc = 16
    sc = wp + ip - op

    new_sc = None
    if sc < min_sc:
      new_sc = min_sc
    elif sc > max_sc:
      new_sc = max_sc

    if new_sc is not None:
      new_wp = new_sc + op - ip
      _set_pos(layer, adjusted_quantize_info, 'w', new_wp)
      logger.debug('Shift cut of layer {} is {}. It exceeds range [{}, {}]. '
                   'Modify wpos from {} to {}.'.format(layer.name, int(sc),
                                                       int(min_sc), int(max_sc),
                                                       int(wp), int(new_wp)))
  return adjusted_quantize_info


def _adjust_shift_bias(model, quantize_info):
  """Adjust the shift bias of layer.

  shift_bias = wp + ip - bp

  DPU compiler constraints of shift_bias:
    1. min(0, -(24 - (8 + shift_cut))) <= shfit_bias <= 16, while shift_cut = wp + ip - op
  """
  adjusted_quantize_info = copy.deepcopy(quantize_info)

  for i in range(1, len(model.layers)):
    layer = model.layers[i]
    if not isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper):
      continue

    # Only adjust shift_cut and shift_bias for Conv-like layers
    if not (isinstance(layer.layer, keras.layers.Conv2D) or
            isinstance(layer.layer, keras.layers.DepthwiseConv2D) or
            isinstance(layer.layer, keras.layers.Conv2DTranspose) or
            isinstance(layer.layer, keras.layers.Dense)):
      continue

    ip, wp, op, bp = _get_iwob_pos(layer, adjusted_quantize_info)

    if isinstance(wp, np.ndarray):
      logger.debug(
          'Not support adjust shift bias for per_channel quantization.')
      return adjusted_quantize_info

    if not _is_valid(ip, wp, op, bp):
      logger.debug('Skip shift bias adjustment for layer {}, '
                   'its quantize pos is [i={}, w={}, b={}, o={}]'.format(
                       layer.name, ip, wp, bp, op))
      return adjusted_quantize_info

    sc = wp + ip - op
    min_sb = min(0, -(24 - (8 + sc)))
    max_sb = 16
    sb = wp + ip - bp

    new_sb = None
    if sb < min_sb:
      new_sb = min_sb
    elif sb > max_sb:
      new_sb = max_sb

    if new_sb is not None:
      new_bp = wp + ip - new_sb
      _set_pos(layer, adjusted_quantize_info, 'b', new_bp)
      logger.debug('Shift bias of layer {} is {}. It exceeds range [{}, {}]. '
                   'Modify bpos from {} to {}.'.format(layer.name, int(sb),
                                                       int(min_sb), int(max_sb),
                                                       int(bp), int(new_bp)))
  return adjusted_quantize_info


def _adjust_vitis_sigmoid(model, quantize_info):
  """Adjust quantize info of VitisSigmoid layers.

  DPU compiler constraints for VitisSigmoid:
    1. input pos of VitisSigmoid >= 0
    2. output pos of VitisSigmoid >= 7
  """
  adjusted_quantize_info = copy.deepcopy(quantize_info)

  for i in range(1, len(model.layers)):
    layer = model.layers[i]
    if isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper) and isinstance(
        layer.layer, vitis_activation.VitisSigmoid):
      pre_layer = layer.inbound_nodes[0].inbound_layers
      ipos = _get_pos(pre_layer, adjusted_quantize_info, 'o')
      if ipos < 0:
        _set_pos(pre_layer, adjusted_quantize_info, 'o', 0)
        logger.debug(
            'Input quantize pos of VitisSimoid layer {} is {}, modify it to 0 '
            'to meet the DPU constraints.'.format(layer.name, int(ipos)))

      opos = _get_pos(layer, adjusted_quantize_info, 'o')
      if opos < 7.0:
        _set_pos(layer, adjusted_quantize_info, 'o', 7.0)
        logger.debug(
            'Output quantize pos of VitisSimoid layer {} is {}, modify it to 7 '
            'to meet the DPU constraints.'.format(layer.name, int(opos)))

  return adjusted_quantize_info


def _adjust_shift_read(model, quantize_info):
  """Adjust the shift bias of layer.

  shift_read = max(ip) - min(ip)

  DPU compiler constraints of shift_bias:
    1. 0 <= shfit_read <= 15
  """
  adjusted_quantize_info = copy.deepcopy(quantize_info)

  for i in range(1, len(model.layers)):
    layer = model.layers[i]
    if not isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper):
      continue

    # Only adjust shift_read for Add and Multiply layers
    if not (isinstance(layer.layer, keras.layers.Add) or
            isinstance(layer.layer, keras.layers.Multiply)):
      continue

    ip_layers = []
    ips = []
    for l in layer.inbound_nodes[0].inbound_layers:
      ip_layers.append(l)
      ips.append(_get_pos(l, quantize_info, 'o'))

    id_max = np.argmax(ips)
    id_min = np.argmin(ips)
    sr = ips[id_max] - ips[id_min]
    min_sr, max_sr = 0, 15

    new_sr = None
    if sr > max_sr:
      new_sr = max_sr

    if new_sr is not None:
      new_ip_max = ips[id_min] + new_sr
      _set_pos(ip_layers[id_max], adjusted_quantize_info, 'o', new_ip_max)
      logger.debug(
          'Shift read of layer {} is {}({}-{}). It exceeds range [{}, {}]. '
          'Modify ipos from {} to {}.'.format(layer.name, int(sr),
                                              int(ips[id_max]),
                                              int(ips[id_min]), int(min_sr),
                                              int(max_sr), int(ips[id_max]),
                                              int(new_ip_max)))
  return adjusted_quantize_info


def adjust_quantize_info(model, quantize_info, adjust_vitis_sigmoid,
                         adjust_shift_cut, adjust_shift_bias,
                         adjust_shift_read):
  """Adjust the quantize info to meet the compiler constraints."""

  adjusted_quantize_info = copy.deepcopy(quantize_info)

  # Adjust VitisSigmoid quantize info
  if adjust_vitis_sigmoid:
    adjusted_quantize_info = _adjust_vitis_sigmoid(model,
                                                   adjusted_quantize_info)

  # Adjust shift_read
  if adjust_shift_read:
    adjusted_quantize_info = _adjust_shift_read(model, adjusted_quantize_info)

  # Adjust shift_cut
  if adjust_shift_cut:
    adjusted_quantize_info = _adjust_shift_cut(model, adjusted_quantize_info)

  # Adjust shift_bias
  if adjust_shift_bias:
    adjusted_quantize_info = _adjust_shift_bias(model, adjusted_quantize_info)
  return adjusted_quantize_info


def _make_quantizer(quantizer_type_name, quantizer_params):
  try:
    quantizer_cls = getattr(vitis_quantizers, quantizer_type_name)
    quantizer = quantizer_cls(**quantizer_params)
  except Exception as e:
    logger.error(
        'Fail to make quantizer `{}` with params `{}`, error: {}'.format(
            quantizer_type_name, quantizer_params, e))
  return quantizer


class ConvertPof2SToFSQuantizeStrategy(transforms.Transform):
  """Convert quantize strategy of quantized layers from pof2s to fs."""

  def __init__(self, use_framework_quant=True):
    self.use_framework_quant = use_framework_quant
    super(ConvertPof2SToFSQuantizeStrategy, self).__init__()

  def pattern(self):
    return LayerPattern('Vitis>VitisQuantize|Vitis>QuantizeWrapper', {}, [])

  def replacement(self, match_layer):
    layer_type = match_layer.layer['class_name']

    if layer_type == 'Vitis>VitisQuantize':
      quantizer = match_layer.layer['config']['quantizer']
      if quantizer['class_name'] == 'Vitis>Pof2SQuantizer':
        pof2s_quantizer = deserialize_keras_object(quantizer)
        fs_quantizer = pof2s_quantizer.convert_to_fs_quantizer(
            self.use_framework_quant)
        quantizer.update(serialize_keras_object(fs_quantizer))
    elif layer_type == 'Vitis>QuantizeWrapper':
      quantize_config = match_layer.layer['config']['quantize_config']
      if not quantize_config['class_name'] == 'Vitis>NoQuantizeConfig':
        config = quantize_config['config']
        quantizers = config['weight_quantizers'] + config[
            'bias_quantizers'] + config['activation_quantizers'] + config[
                'output_quantizers']
        for quantizer in quantizers:
          if quantizer['quantizer_type'] == 'Pof2SQuantizer':
            pof2s_quantizer = _make_quantizer(quantizer['quantizer_type'],
                                              quantizer['quantizer_params'])
            fs_quantizer = pof2s_quantizer.convert_to_fs_quantizer(
                self.use_framework_quant)
            quantizer.update({
                'quantizer_type': 'FSQuantizer',
                'quantizer_params': fs_quantizer.get_config()
            })

    match_layer.weights = self._convert_weights(match_layer.weights)
    return match_layer

  def _convert_weights(self, weights):
    """Helper function to convert weights from TQTQuantizer to Pof2SQuantizer."""
    new_weights = collections.OrderedDict(weights)
    for k, v in weights.items():
      if k.endswith('_pos:0'):
        min_key = k[:-6] + '_min:0'
        max_key = k[:-6] + '_max:0'
        new_weights[min_key] = -128. / np.power(2, v)
        new_weights[max_key] = 127. / np.power(2, v)
        new_weights.pop(k)
    return new_weights

  def custom_objects(self):
    return {}
