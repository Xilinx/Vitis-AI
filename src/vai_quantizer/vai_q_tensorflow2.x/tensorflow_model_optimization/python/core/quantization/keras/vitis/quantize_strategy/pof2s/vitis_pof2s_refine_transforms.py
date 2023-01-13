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
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_custom_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_configs
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantizers
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_quantize as vitis_quantize_layer
from tensorflow_model_optimization.python.core.quantization.keras.vitis.layers import vitis_activation
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

serialize_keras_object = tf.keras.utils.serialize_keras_object
deserialize_keras_object = tf.keras.utils.deserialize_keras_object
LayerPattern = transforms.LayerPattern
logger = common_utils.VAILogger
keras = tf.keras


class QuantPosManager(object):
  """The helper class to manage the quantize posistions."""

  def __init__(self, model, quantize_info):
    self.model = model
    self.quantize_info = copy.deepcopy(quantize_info)

  def get_pos(self, layer, key):
    """Get the quantize pos of layer:key in quantize_info."""
    if isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper):
      q_info = self.quantize_info[layer.layer.name]

      if not q_info:
        return None

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
        q_info = self.quantize_info[layer.name]
        return q_info['info']['quant_pos_var']
    elif isinstance(layer, vitis_custom_wrapper.CustomOpWrapper) and isinstance(
        layer.layer, vitis_quantize_wrapper.QuantizeWrapper):
      if key == 'o':
        q_info = self.quantize_info[layer.layer.layer.name]
        return q_info['output_0']['info']['quant_pos_var']
    else:
      logger.debug(
          'Fail to get quantize position for layer {}(key:{}), this may happen when it is not quantized.'
          .format(layer.name, key))
      return None

  def set_pos(self, layer, key, new_pos):
    """Set the quantize pos of layer:key in quantize_info to new_pos."""
    if isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper):
      q_info = self.quantize_info[layer.layer.name]
      for k, v in q_info.items():
        if k == 'NoQuantizeActivation':
          continue
        if key == 'w' and v.get('type') == 'weight' and k.endswith('kernel:0'):
          v['info']['quant_pos_var'] = new_pos
          return True
        elif key == 'b' and v.get('type') == 'weight' and k.endswith('bias:0'):
          v['info']['quant_pos_var'] = new_pos
          return True
        elif key == 'o' and v.get('type') in [
            'post_activation', 'pre_activation', 'output'
        ]:
          v['info']['quant_pos_var'] = new_pos
          return True
    elif isinstance(layer, vitis_quantize_layer.VitisQuantize):
      if key == 'o':
        q_info = self.quantize_info[layer.name]
        q_info['info']['quant_pos_var'] = new_pos
        return True

    logger.debug("Fail to set pos for layer {}(key:{}, new_pos:{}).".format(
        layer.name, key, new_pos))
    return False

  def get_ipos(self, layer, input_id=None):
    """Get the input quantize position of layer in quantize_info."""
    pre_layers = layer.inbound_nodes[0].inbound_layers

    if not pre_layers:
      logger.debug('Layer {} has 0 inputs, fail to get_ipos() for it.'.format(
          layer.name))
      return None

    if input_id is not None and not isinstance(pre_layers, list):
      if not input_id == 0:
        logger.debug(
            'Layer {} has only 1 input, expect input_id to be None or 0, but got {}.'
            .format(layer.name, input_id))
        return None

    if input_id is None and isinstance(pre_layers, list):
      logger.debug(
          'Layer {} has {} inputs, please specify input_id for get_ipos().'
          .format(layer.name, len(pre_layers)))
      return None

    pre_layer = pre_layers[input_id] if isinstance(pre_layers,
                                                   list) else pre_layers
    ipos = self.get_pos(pre_layer, 'o')

    def _is_transparent_layer(layer):
      # Recursive searching for layers with empty quantize info to skip some
      # special layers which are transparent to quantization, such as Reshape,
      # Flatten layers.
      _TRANSPARENT_LAYERS = (tf.keras.layers.Reshape, tf.keras.layers.Flatten,
                             tf.keras.layers.ZeroPadding2D)
      return isinstance(pre_layer,
                        vitis_quantize_wrapper.QuantizeWrapper) and isinstance(
                            pre_layer.layer, _TRANSPARENT_LAYERS)

    if ipos is None:
      if _is_transparent_layer(pre_layer):
        ipos, pre_layer = self.get_ipos(pre_layer)
      else:
        logger.debug(
            'Layer {}\'s input layer {} is not quantized, fail to get_ipos() for it.'
            .format(layer.name, pre_layer.name))

    return ipos, pre_layer

  def get_opos(self, layer):
    """Get the output quantize position of layer in quantize_info."""
    if not isinstance(layer, (vitis_quantize_wrapper.QuantizeWrapper,
                              vitis_quantize_layer.VitisQuantize)):
      logger.warning('Layer {} is not quantized, fail to get its opos.'.format(
          layer.name))
      return None

    if isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper) and hasattr(
        layer.layer, 'activation') and isinstance(
            layer.layer.activation.activation,
            vitis_quantize_aware_activation.NoQuantizeActivation):
      post_layer = layer.outbound_nodes[0].outbound_layer
      opos = self.get_pos(post_layer, 'o')
      return opos, post_layer
    elif isinstance(layer.quantize_config,
                    vitis_quantize_configs.NoQuantizeConfig):
      post_layer = layer.outbound_nodes[0].outbound_layer
      opos = self.get_pos(post_layer, 'o')
      return opos, post_layer
    else:
      opos = self.get_pos(layer, 'o')
      return opos, layer

  def get_wpos(self, layer):
    """Get the weights quantize position of layer in quantize_info."""
    wpos = self.get_pos(layer, 'w')
    return wpos

  def get_bpos(self, layer):
    """Get the biases quantize position of layer in quantize_info."""
    bpos = self.get_pos(layer, 'b')
    return bpos

  def is_valid(self, pos):
    """Check if the quantize pos is valid."""
    if pos is None:
      return False
    if isinstance(pos, np.ndarray) and None in pos:
      return False
    return True

  def adjust_shift_cut(self):
    """Adjust the shift cut of layers.

    shift_cut = wpos + ipos - opos

    DPU compiler constraints of shift_cut:
      1. 0 <= shift_cut <= 16
    """
    for i, layer in enumerate(self.model.layers):
      if not isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper):
        continue

      # Only adjust shift_cut and shift_bias for Conv-like layers
      if not (isinstance(layer.layer, keras.layers.Conv2D) or
              isinstance(layer.layer, keras.layers.DepthwiseConv2D) or
              isinstance(layer.layer, keras.layers.Conv2DTranspose) or
              isinstance(layer.layer, keras.layers.Dense)):
        continue

      ipos, _ = self.get_ipos(layer)
      opos, _ = self.get_opos(layer)
      wpos = self.get_wpos(layer)

      if any([isinstance(pos, np.ndarray) for pos in [ipos, opos, wpos]]):
        logger.debug(
            'Adjust shift cut for per_channel quantization is not supported.')
        continue

      if not all([self.is_valid(p) for p in [ipos, wpos, opos]]):
        logger.debug('Skip shift cut adjustment for layer {}, '
                     'its quantize pos is [i={}, w={}, o={}]'.format(
                         layer.name, ipos, wpos, opos))
        continue

      # Adjust shift_cut
      min_sc = 0
      max_sc = 16
      sc = wpos + ipos - opos

      new_sc = None
      if sc < min_sc:
        new_sc = min_sc
      elif sc > max_sc:
        new_sc = max_sc

      if new_sc is not None:
        new_wpos = new_sc + opos - ipos
        self.set_pos(layer, 'w', new_wpos)
        logger.debug('Shift cut of layer {} is {}. It exceeds range [{}, {}]. '
                     'Modify wpos from {} to {}.'.format(
                         layer.name, int(sc), int(min_sc), int(max_sc),
                         int(wpos), int(new_wpos)))

  def adjust_shift_bias_leakyrelu(self):
    """Adjust the shift bias of previous leakyrelu layer .

    shift_bias_leakyrelu = wpos + ipos - bpos

    DPU compiler constraints of shift_bias before leakyrelu:
      1. 0 <= shfit_bias <= 16
    """
    for i, act_layer in enumerate(self.model.layers):
      if not isinstance(act_layer, vitis_quantize_wrapper.QuantizeWrapper):
        continue

      # Only adjust shift_cut and shift_bias for Conv-like layers

      if not (isinstance(act_layer.layer, keras.layers.LeakyReLU)):
        continue
      layer = act_layer.inbound_nodes[0].inbound_layers

      if not (isinstance(layer.layer, keras.layers.Conv2D) or
              isinstance(layer.layer, keras.layers.DepthwiseConv2D) or
              isinstance(layer.layer, keras.layers.Conv2DTranspose) or
              isinstance(layer.layer, keras.layers.Dense)):
        continue

      ipos, _ = self.get_ipos(layer)
      opos, _ = self.get_opos(layer)
      wpos = self.get_wpos(layer)
      bpos = self.get_bpos(layer)

      if any([isinstance(pos, np.ndarray) for pos in [ipos, opos, wpos, bpos]]):
        logger.debug(
            'Adjust shift bias for per_channel quantization is not supported.')
        continue

      if not all([self.is_valid(p) for p in [ipos, wpos, bpos, opos]]):
        logger.debug('Skip shift bias adjustment for layer {}, '
                     'its quantize pos is [i={}, w={}, b={}, o={}]'.format(
                         layer.name, ipos, wpos, bpos, opos))
        continue

      # Adjust shift_bias
      min_sb = 0
      max_sb = 16
      shift_bias = wpos + ipos - bpos

      new_sb = None
      if shift_bias < min_sb:
        new_sb = min_sb
      elif shift_bias > max_sb:
        new_sb = max_sb

      if new_sb is not None:
        new_bpos = wpos + ipos - new_sb
        self.set_pos(layer, 'b', new_bpos)
        logger.debug(
            'Shift bias leakyrelu of layer {} is {}. It exceeds range [{}, {}]. '
            'Modify bpos from {} to {}.'.format(layer.name, int(shift_bias),
                                                int(min_sb), int(max_sb),
                                                int(bpos), int(new_bpos)))

  def adjust_shift_bias(self):
    """Adjust the shift bias of layer.

    shift_bias = wpos + ipos - bpos

    DPU compiler constraints of shift_bias:
      1. min(0, -(24 - (8 + shift_cut))) <= shfit_bias <= 16
    """
    for i, layer in enumerate(self.model.layers):
      if not isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper):
        continue

      # Only adjust shift_cut and shift_bias for Conv-like layers
      if not (isinstance(layer.layer, keras.layers.Conv2D) or
              isinstance(layer.layer, keras.layers.DepthwiseConv2D) or
              isinstance(layer.layer, keras.layers.Conv2DTranspose) or
              isinstance(layer.layer, keras.layers.Dense)):
        continue

      ipos, _ = self.get_ipos(layer)
      opos, _ = self.get_opos(layer)
      wpos = self.get_wpos(layer)
      bpos = self.get_bpos(layer)

      if any([isinstance(pos, np.ndarray) for pos in [ipos, opos, wpos, bpos]]):
        logger.debug(
            'Adjust shift bias for per_channel quantization is not supported.')
        continue

      if not all([self.is_valid(p) for p in [ipos, wpos, bpos, opos]]):
        logger.debug('Skip shift bias adjustment for layer {}, '
                     'its quantize pos is [i={}, w={}, b={}, o={}]'.format(
                         layer.name, ipos, wpos, bpos, opos))
        continue

      # Adjust shift_bias
      shift_cut = wpos + ipos - opos
      min_sb = min(0, -(24 - (8 + shift_cut)))
      max_sb = 16
      shift_bias = wpos + ipos - bpos

      new_sb = None
      if shift_bias < min_sb:
        new_sb = min_sb
      elif shift_bias > max_sb:
        new_sb = max_sb

      if new_sb is not None:
        new_bpos = wpos + ipos - new_sb
        self.set_pos(layer, 'b', new_bpos)
        logger.debug('Shift bias of layer {} is {}. It exceeds range [{}, {}]. '
                     'Modify bpos from {} to {}.'.format(
                         layer.name, int(shift_bias), int(min_sb), int(max_sb),
                         int(bpos), int(new_bpos)))

  def adjust_vitis_sigmoid(self):
    """Adjust quantize info of VitisSigmoid layers.

    DPU compiler constraints for VitisSigmoid:
      1. input pos of VitisSigmoid >= 0 && <= 15
      2. output pos of VitisSigmoid >= 7
      3. shift_sigmoid >= 0 && shift_sigmoid <= 31 where
         shift_sigmoid = 14 + 'input pos' - ' output pos'
    """
    for i, layer in enumerate(self.model.layers):
      if isinstance(layer,
                    vitis_quantize_wrapper.QuantizeWrapper) and isinstance(
                        layer.layer, vitis_activation.VitisSigmoid):
        ipos, ipos_layer = self.get_ipos(layer)
        opos, opos_layer = self.get_opos(layer)

        new_ipos = ipos if ipos > 0 else 0
        new_ipos = new_ipos if new_ipos <= 15 else 15

        new_opos = opos if opos > 7 else 7
        shift_sigmoid = 14 + new_ipos - new_opos  # will not bigger than 31 now
        new_opos = new_opos if shift_sigmoid > 0 else 14 + new_ipos

        if new_ipos != ipos:
          self.set_pos(ipos_layer, 'o', new_ipos)
          logger.debug(
              'Input quantize pos of VitisSimoid layer {} is {}, modify it to {} '
              'to meet the DPU constraints.'.format(layer.name, int(ipos),
                                                    int(new_ipos)))

        if new_opos != opos:
          self.set_pos(opos_layer, 'o', new_opos)
          logger.debug(
              'Output quantize pos of VitisSimoid layer {} is {}, modify it to {} '
              'to meet the DPU constraints.'.format(layer.name, int(opos),
                                                    int(new_opos)))

  def adjust_shift_read(self):
    """Adjust the shift bias of layer.

    shift_read = max(ipos) - min(ipos)

    DPU compiler constraints of shift_bias:
      1. 0 <= shift_read <= 15
    """

    for i, layer in enumerate(self.model.layers):
      if not isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper):
        continue

      # Only adjust shift_read for Add and Multiply layers
      if not (isinstance(layer.layer, keras.layers.Add) or
              isinstance(layer.layer, keras.layers.Multiply)):
        continue

      ipos_layers = []
      iposes = []
      skip = False
      for i, l in enumerate(layer.inbound_nodes[0].inbound_layers):
        ipos, ipos_layer = self.get_ipos(layer, i)
        ipos_layers.append(ipos_layer)
        if ipos is None:
          logger.debug(
              'Fail to get quantize position for layer {}(input:{}) (output of layer {}), '
              'skip adjust_shift_read for it.'.format(layer.name, i, l.name))
          skip = True
        iposes.append(ipos)

      if skip:
        continue

      id_max = np.argmax(iposes)
      id_min = np.argmin(iposes)
      sr = iposes[id_max] - iposes[id_min]
      min_sr, max_sr = 0, 15

      new_sr = None
      if sr > max_sr:
        new_sr = max_sr

      if new_sr is not None:
        new_ipos_max = iposes[id_min] + new_sr
        self.set_pos(ipos_layers[id_max], 'o', new_ipos_max)
        logger.debug(
            'Shift read of layer {} is {}({}-{}). It exceeds range [{}, {}]. '
            'Modify ipos from {} to {}.'.format(layer.name, int(sr),
                                                int(iposes[id_max]),
                                                int(iposes[id_min]),
                                                int(min_sr), int(max_sr),
                                                int(iposes[id_max]),
                                                int(new_ipos_max)))

  def adjust_shift_write(self):
    """Adjust the shift write of layer.

    shift_write = min(ipos) - opos

    DPU compiler constraints of shift_write:
      1. -15 <= shift_write <= 15
    """
    for i, layer in enumerate(self.model.layers):
      if not isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper):
        continue

      # Only adjust shift_write for Add and Multiply layers
      if not (isinstance(layer.layer, keras.layers.Add) or
              isinstance(layer.layer, keras.layers.Multiply)):
        continue

      ipos_layers = []
      iposes = []
      skip = False
      for i, l in enumerate(layer.inbound_nodes[0].inbound_layers):
        ipos, ipos_layer = self.get_ipos(layer, i)
        ipos_layers.append(ipos_layer)
        if ipos is None:
          logger.debug(
              'Fail to get quantize position for layer {}(input:{}) (output of layer {}), '
              'skip adjust_shift_write for it.'.format(layer.name, i, l.name))
          skip = True
        else:
          iposes.append(ipos)

      opos, opos_layer = self.get_opos(layer)
      if opos is None:
        logger.debug('Fail to get quantize position for layer {}(output:0), '
                     'skip adjust_shift_write for it.'.format(layer.name))
        skip = True

      if skip:
        continue

      # Adjust shift_write
      id_min = np.argmin(iposes)
      id_max = np.argmax(iposes)

      sw = iposes[id_min] - opos
      min_sw, max_sw = -15, 15

      new_sw = None
      if sw > max_sw:
        new_sw = iposes[id_min] - max_sw
      elif sw < min_sw:
        new_sw = iposes[id_min] - min_sw

      if new_sw is not None:
        new_opos = iposes[id_min] - new_sw
        self.set_pos(opos_layer, 'o', new_opos)
        logger.debug(
            'Shift write of layer {} is {}({}-{}). It exceeds range [{}, {}]. '
            'Modify opos from {} to {}.'.format(layer.name, int(sw),
                                                int(iposes[id_min]), int(opos),
                                                int(min_sw), int(max_sw),
                                                int(opos), int(new_opos)))

  def adjust_shift_swish(self):
    """Adjust the shift of Swish layer's Multiply op.

    shift_swish = 'input 0 pos' + 'input 1 pos' - 'output pos'

    DPU compiler constraints of shift_swish:
      1. 0 <= shift_swish <= 15
    """

    def _be_sigmoid_layer(layer):
      '''
      it's a swish's sigmoid layer or not
      '''
      if not (isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper) and
              isinstance(layer.layer, vitis_activation.VitisSigmoid)):
        return False
      elif isinstance(layer.inbound_nodes[0].inbound_layers, list):
        return False
      elif not layer.inbound_nodes[0].inbound_layers:
        return False
      else:
        return True

    def _belong_to_swish(layer0, layer1):
      '''
      swish = mul(x, sigmoid(x))
      so one is sigmoid and another is x
      '''
      if (_be_sigmoid_layer(layer0) and not _be_sigmoid_layer(layer1)):
        return layer0.inbound_nodes[0].inbound_layers.name == layer1.name
      elif (not _be_sigmoid_layer(layer0) and _be_sigmoid_layer(layer1)):
        return layer0.name == layer1.inbound_nodes[0].inbound_layers.name
      else:
        return False

    for i, layer in enumerate(self.model.layers):
      if not isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper):
        continue

      # Only adjust shift_write for Multiply layers
      if not (isinstance(layer.layer, keras.layers.Multiply)):
        continue

      ipos1, ipos_layer1 = self.get_ipos(layer, 1)
      if (ipos1 == None or ipos_layer1 == None):
        logger.debug('Fail to get quantize position for layer {}(input:{}), '
                     'skip adjust_shift_swish for it.'.format(layer.name, 1))
        continue

      ipos0, ipos_layer0 = self.get_ipos(layer, 0)
      if (ipos0 == None or ipos_layer0 == None):
        logger.debug('Fail to get quantize position for layer {}(input:{}), '
                     'skip adjust_shift_swish for it.'.format(layer.name, 0))
        continue

      # Comfirm it's a swish's mul layer or not
      if not _belong_to_swish(ipos_layer0, ipos_layer1):
        continue

      opos, opos_layer = self.get_opos(layer)
      if opos is None:
        logger.debug('Fail to get quantize position for layer {}(output:0), '
                     'skip adjust_shift_swish for it.'.format(layer.name))
        continue

      # Adjust shift_swish
      min_sh, max_sh = 0, 15

      shift_swish = ipos0 + ipos1 - opos

      new_opos = opos
      if (shift_swish < min_sh):
        new_opos = ipos0 + ipos1 - min_sh
      elif (shift_swish > max_sh):
        new_opos = ipos0 + ipos1 - max_sh

      if new_opos != opos:
        self.set_pos(layer, 'o', new_opos)
        logger.debug(
            'Shift Swish of layer {} is {}({}+{}-{}). It exceeds range [{}, {}]. '
            'Modify opos from {} to {}.'.format(layer.name, int(shift_swish),
                                                int(ipos0), int(ipos1),
                                                int(opos), int(min_sh),
                                                int(max_sh), int(opos),
                                                int(new_opos)))

  def align_concat(self):
    """Align concat op's inputs and output pos.
    """
    for layer in self.model.layers:
      if (isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper) and
          isinstance(layer.layer, keras.layers.Concatenate)):
        input_layers_num = len(layer.inbound_nodes[0].inbound_layers)
        opos, opos_layer = self.get_opos(layer)
        min_pos = opos
        for i in range(input_layers_num):
          ipos, _ = self.get_ipos(layer, i)
          if ipos is not None:
            min_pos = min(ipos, min_pos)
        if opos != min_pos:
          self.set_pos(opos_layer, 'o', min_pos)
          logger.debug('Output pos of concat layer {} is {}, min_pos is {}. '
                       'Modify opos from {} to {}.'.format(
                           layer.name, int(opos), int(min_pos), int(opos),
                           int(min_pos)))

        for i in range(input_layers_num):
          ipos, ipos_layer = self.get_ipos(layer, i)
          if ipos is not None and ipos != min_pos:
            self.set_pos(ipos_layer, 'o', min_pos)
            logger.debug('Input pos of concat layer {} is {}, min_pos is {}. '
                         'Modify ipos from {} to {}.'.format(
                             layer.name, int(ipos), int(min_pos), int(ipos),
                             int(min_pos)))

  def align_pool(self):
    """Align max/avg pooling input and output pos.
    """
    for layer in self.model.layers:
      if (isinstance(layer, vitis_quantize_wrapper.QuantizeWrapper) and
          (isinstance(layer.layer, keras.layers.MaxPooling2D) or
           isinstance(layer.layer, keras.layers.AveragePooling2D))):
        opos, opos_layer = self.get_opos(layer)
        ipos, ipos_layer = self.get_ipos(layer)
        if ipos is not None and opos > ipos:
          self.set_pos(opos_layer, 'o', ipos)
          logger.debug(
              'Input pos of pooling layer {} is {}. Output pos of pooling layer {} is {}.'
              'Modify opos from {} to {}.'.format(layer.name, int(ipos),
                                                  layer.name, int(opos),
                                                  int(opos), int(ipos)))
        elif ipos is not None and opos < ipos:
          self.set_pos(ipos_layer, 'o', opos)
          logger.debug(
              'Input pos of pooling layer {} is {}. Output pos of pooling layer {} is {}.'
              'Modify ipos from {} to {}.'.format(layer.name, int(ipos),
                                                  layer.name, int(opos),
                                                  int(ipos), int(opos)))


def adjust_quantize_info(model,
                         quantize_info,
                         adjust_vitis_sigmoid,
                         adjust_shift_cut,
                         adjust_shift_bias,
                         adjust_shift_read,
                         adjust_shift_write,
                         adjust_shift_swish,
                         align_concat,
                         align_pool,
                         adjust_shift_bias_leakyrelu=False):
  """Adjust the quantize info to meet the compiler constraints."""

  manager = QuantPosManager(model, quantize_info)

  if adjust_vitis_sigmoid:
    manager.adjust_vitis_sigmoid()

  if adjust_shift_read:
    manager.adjust_shift_read()

  if adjust_shift_write:
    manager.adjust_shift_write()

  if adjust_shift_swish:
    manager.adjust_shift_swish()

  if adjust_shift_cut:
    manager.adjust_shift_cut()

  if adjust_shift_bias:
    manager.adjust_shift_bias()

  if adjust_shift_bias_leakyrelu:
    manager.adjust_shift_bias_leakyrelu()

  if align_concat:
    manager.align_concat()

  if align_pool:
    manager.align_pool()
  return manager.quantize_info


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
