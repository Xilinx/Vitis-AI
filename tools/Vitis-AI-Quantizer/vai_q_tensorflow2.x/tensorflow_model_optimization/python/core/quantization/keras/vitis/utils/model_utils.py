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
"""Model Utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import copy
import collections
import pprint

from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_quantize as vitis_quantize_layer
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.graph_transformations import model_transformer
from tensorflow_model_optimization.python.core.quantization.keras.vitis.optimizations import vitis_optimize_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.eight_bit import vitis_8bit_quantize_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

logger = common_utils.VAILogger
keras = tf.keras


def remove_layer(model, class_name, name='.*'):
  """Remove given layer from the model."""
  transforms = [vitis_optimize_transforms.RemoveLayer(class_name, name)]
  transformed_model, _ = model_transformer.ModelTransformer(
      model, transforms, None, None).recursive_transform()
  return transformed_model


def convert_quantize_strategy(model, conversion='8bit_tqt_to_8bit'):
  transforms = [vitis_optimize_transforms.ConvertQuantizeStrategy(conversion)]
  transformed_model, _ = model_transformer.ModelTransformer(
      model, transforms, None, None).recursive_transform()
  return transformed_model


def separate_conv_act(model):
  transforms = [vitis_optimize_transforms.SeparateConvAct()]
  transformed_model, _ = model_transformer.ModelTransformer(
      model, transforms, None, None).recursive_transform()
  return transformed_model


def fake_conv_bn_fold(model):
  transforms = [vitis_optimize_transforms.FakeConvBNFold()]
  transformed_model, _ = model_transformer.ModelTransformer(
      model, transforms, None, None).recursive_transform()
  return transformed_model


def real_conv_bn_fold(model):
  transforms = [vitis_optimize_transforms.RealConvBNFold()]
  transformed_model, _ = model_transformer.ModelTransformer(
      model, transforms, None, None).recursive_transform()
  return transformed_model


def conv_bn_quantize_fold(model):
  transforms = [vitis_8bit_quantize_transforms.ConvBNQuantizeFold()]
  transformed_model, _ = model_transformer.ModelTransformer(
      model, transforms, None, None).recursive_transform()
  return transformed_model


def replace_hard_sigmoid(model):
  transforms = [vitis_8bit_quantize_transforms.ReplaceHardSigmoid()]
  transformed_model, _ = model_transformer.ModelTransformer(
      model, transforms, None, None).recursive_transform()
  return transformed_model


def get_quantize_info(model):
  """Get the quantize info of the model"""
  quantize_info = collections.OrderedDict()
  for layer in model.layers:
    if isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper):
      quantize_info[layer.layer.name] = layer.get_quantize_info()
    elif isinstance(layer, vitis_quantize_layer.VitisQuantize):
      quantize_info[layer.name] = layer.get_quantize_info()
  return quantize_info


def set_quantize_info(model, new_quantize_info):
  """Set the quantize info of the model"""
  for layer in model.layers:
    if isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper
                 ) and layer.layer.name in new_quantize_info:
      layer.set_quantize_info(new_quantize_info[layer.layer.name])
    elif isinstance(
        layer,
        vitis_quantize_layer.VitisQuantize) and layer.name in new_quantize_info:
      layer.set_quantize_info(new_quantize_info[layer.name])
  return


def save_quantize_info(quantize_info, output_dir='./'):
  """Save the quantize info to the disk."""
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  filename = os.path.join(output_dir, 'quantize_info.txt')
  with open(filename, 'w') as f:
    idx = 0
    for k, v in quantize_info.items():
      f.write(str(idx) + ' ' + k + '\n')
      formatted = pprint.pformat(v)
      for line in formatted.splitlines():
        f.write('  ' + line + '\n')
      idx += 1
  logger.debug(filename + ' saved.')
  return


def save_model(model, filename, output_dir='./'):
  """Save the model to the disk."""
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  filename = os.path.join(output_dir, filename)
  model.save(filename)
  logger.debug(filename + ' saved.')
  return


def is_quantize_layer(layer):
  """Check if QuantizeWrapper or VitisQuantize Layer."""
  return isinstance(layer,
                    vitis_quantize_wrapper.QuantizeWrapper) or isinstance(
                        layer, vitis_quantize_layer.VitisQuantize)


def set_layer_mode(model, mode, layer_names=None):
  """Set the mode of QuantizeWrapper and VitisQuantize Layer."""
  for layer in model.layers:
    if is_quantize_layer(layer):
      if not layer_names or layer.name in layer_names:
        layer.mode = mode
  return


def clone_model_with_weights(model_to_clone):
  """Clone keras model with weights."""
  cloned_model = keras.models.clone_model(model_to_clone)
  cloned_model.set_weights(model_to_clone.get_weights())
  return cloned_model


def dump_model_weights(model, dump_float, output_dir):
  """Dump model weights."""
  # Get weight quantize info
  w_q_map = {}
  for layer in model.layers:
    if is_quantize_layer(layer):
      if isinstance(layer, vitis_quantize_layer.VitisQuantize):
        continue
      layer_quantize_info = layer.get_quantize_info()
      for name, value in layer_quantize_info.items():
        if value.get('type') == 'weight':
          w_name = name.rstrip(':0')
          w_q_map[w_name] = value['info']['quant_pos_var']

  logger.info("Dumping weights/biases...")
  dump_folder = os.path.join(output_dir, "dump_results_weights")
  if not os.path.exists(dump_folder):
    os.makedirs(dump_folder)

  index = 0
  for w in model.weights:
    w_name = w.name.rstrip(':0')
    if w_name not in w_q_map:
      continue

    index = index + 1
    filename = os.path.join(dump_folder, w_name.replace('/', '_'))
    logger.info("Dumping ({}/{}): {}".format(index, len(w_q_map), w_name))

    res = w.numpy()
    res = res.flatten()
    if dump_float:
      res.tofile(filename + '_float.bin')
      np.savetxt(filename + "_float.txt", res, fmt="%s", delimiter=",")

    if w_name in w_q_map:
      res = np.round(res * 2**w_q_map[w_name])
      res = res.clip(-128, 127)
      res.astype(np.int8).tofile(filename + ".bin")
      np.savetxt(
          filename + ".txt", res.astype(np.int8), fmt="%s", delimiter=",")


def dump_model_activations(model, dataset, dump_float, output_dir):
  """Dump model activation."""
  # Get activation quantize info
  a_q_map = {}
  for layer in model.layers:
    if is_quantize_layer(layer):
      layer_quantize_info = layer.get_quantize_info()
      if isinstance(layer, vitis_quantize_layer.VitisQuantize):
        a_q_map[layer.name] = layer_quantize_info['info']['quant_pos_var']
      else:
        for name, value in layer_quantize_info.items():
          if value.get('type') in [
              'output', 'pre_activation', 'post_activation'
          ]:
            a_q_map[layer.name] = value['info']['quant_pos_var']

  if dump_float:
    quant_layers = model.layers
  else:
    quant_layers = [layer for layer in model.layers if is_quantize_layer(layer)]

  model = keras.Model(
      inputs=model.inputs, outputs=[layer.output for layer in quant_layers])

  logger.info("Dumping activations...")
  # TODO: Support dump for multi-batches
  dump_folder = os.path.join(output_dir, "dump_results_0")
  if not os.path.exists(dump_folder):
    os.makedirs(dump_folder)

  results = model.predict(dataset, steps=1)

  index = 0
  for layer, res in zip(quant_layers, results):
    index = index + 1
    a_name = layer.name
    filename = os.path.join(dump_folder, a_name.replace('/', '_'))
    logger.info("Dumping ({}/{}): {}".format(index, len(quant_layers), a_name))
    res = res.flatten()

    if dump_float:
      res.tofile(filename + '_float.bin')
      np.savetxt(filename + "_float.txt", res, fmt="%s", delimiter=",")

    if a_name in a_q_map:
      if isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper) and any(
          act._should_pre_quantize()
          for act in layer._quantize_activations) and not dump_float:
        res.tofile(filename + '_float.bin')
        np.savetxt(filename + "_float.txt", res, fmt="%s", delimiter=",")
      else:
        res = res * 2**a_q_map[a_name]
        res.astype(np.int8).tofile(filename + ".bin")
        np.savetxt(
            filename + ".txt", res.astype(np.int8), fmt="%s", delimiter=",")


def post_quant_adjust(model, quantize_info, adjust_shift_cut,
                      adjust_shift_bias):
  """Adjust the quantize info to meet the compiler constraints."""

  def _get_pos(layer, quantize_info, key):
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

  adjusted_quantize_info = copy.deepcopy(quantize_info)

  # VitisSigmoid adjustment
  for i in range(1, len(model.layers)):
    layer = model.layers[i]
    if isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper) and isinstance(
        layer.layer, vitis_activation.VitisSigmoid):
      opos = _get_pos(layer, adjusted_quantize_info, 'o')
      if opos < 7.0:
        _set_pos(layer, adjusted_quantize_info, 'o', 7.0)
        logger.debug(
            'Quantize pos of VitisSimoid layer {} is {}, modify it to 7 '
            'to meet the DPU constraints.'.format(layer.name, int(opos)))

  # Adjust shift_cut and shift_bias
  if not adjust_shift_cut and not adjust_shift_bias:
    return adjusted_quantize_info

  def _adjust_shift_cut(layer, adjusted_quantize_info, ip, wp, bp, op):
    min_sc = 0
    max_sc = 16
    sc = wp + ip - op

    new_sc = None
    if sc < min_sc:
      new_sc = min_sc
    elif sc > max_sc:
      new_sc = max_sc

    if new_sc:
      new_wp = min_sc + op - ip
      _set_pos(layer, adjusted_quantize_info, 'w', new_wp)
      logger.debug('Shift cut of layer {} is {}. It exceeds range [{}, {}]. '
                   'Modify wpos from {} to {}.'.format(layer.name, int(sc),
                                                       int(min_sc), int(max_sc),
                                                       int(wp), int(new_wp)))

  def _adjust_shift_bias(layer, adjusted_quantize_info, ip, wp, bp, op):
    sc = wp + ip - op
    min_sb = min(0, -(24 - (8 + sc)))
    max_sb = 16
    sb = wp + ip - bp

    new_sb = None
    if sb < min_sb:
      new_sb = min_sb
    elif sb > max_sb:
      new_sb = max_sb

    if new_sb:
      new_bp = wp + ip - new_sb
      _set_pos(layer, adjusted_quantize_info, 'b', new_bp)
      logger.debug('Shift bias of layer {} is {}. It exceeds range [{}, {}]. '
                   'Modify bpos from {} to {}.'.format(layer.name, int(sb),
                                                       int(min_sb), int(max_sb),
                                                       int(bp), int(new_bp)))

  for i in range(1, len(model.layers)):
    layer = model.layers[i]
    if not isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper):
      continue

    if not (isinstance(layer.layer, keras.layers.Conv2D) or
            isinstance(layer.layer, keras.layers.DepthwiseConv2D) or
            isinstance(layer.layer, keras.layers.Conv2DTranspose) or
            isinstance(layer.layer, keras.layers.Dense)):
      continue

    wp = _get_pos(layer, quantize_info, 'w')
    bp = _get_pos(layer, quantize_info, 'b')

    if isinstance(layer.layer.activation.activation,
                  vitis_quantize_aware_activation.NoQuantizeActivation):
      post_layer = layer.outbound_nodes[0].outbound_layer
      op = _get_pos(post_layer, quantize_info, 'o')
    else:
      op = _get_pos(layer, quantize_info, 'o')

    pre_layer = layer.inbound_nodes[0].inbound_layers
    ip = _get_pos(pre_layer, quantize_info, 'o')

    if adjust_shift_cut and None not in [ip, wp, op]:
      _adjust_shift_cut(layer, adjusted_quantize_info, ip, wp, bp, op)
    else:
      logger.debug('Skip shift cut adjustment for layer {}, '
                   'its quantize pos is [i={}, w={}, b={}, o={}]'.format(
                       layer.name, ip, wp, bp, op))

    if adjust_shift_bias and None not in [ip, wp, bp, op]:
      _adjust_shift_bias(layer, adjusted_quantize_info, ip, wp, bp, op)
    else:
      logger.debug('Skip shift bias adjustment for layer {}, '
                   'its quantize pos is [i={}, w={}, b={}, o={}]'.format(
                       layer.name, ip, wp, bp, op))
  return adjusted_quantize_info
