

#
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
#

import copy
import numpy as np
import math
from nndct_shared.base import NNDCT_KEYS, GLOBAL_MAP, NNDCT_DEBUG_LVL, NNDCT_OP
from nndct_shared.algorithms import breadth_first_search_handler
from nndct_shared.nndct_graph import NndctGraphHolder, Tensor
from nndct_shared import utils as nndct_utils
from .quant_ops import normal_quant_neuron

def quantize_data2int(data, bn, fp, method=2):
  return normal_quant_neuron(
      data, maxamps=[[2**(bn - 1)], [2**fp]], round_method=method, as_int=True)

def maybe_get_quantizer(quantizer=None):
  quantizer = quantizer or GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANTIZER)
  if quantizer:
    return quantizer.quant_mode, quantizer
  else:
    return GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_MODE), None

def is_quant_end_point(graph, node, quant_types):
  if len(graph.parents(node.name)) == 0:
    return False
  __QuantNodes = []

  def __check_end(node_name):
    if graph.node(node_name).op.type in quant_types:
      __QuantNodes.append(node_name)

  def __children_names(node_name):
    for c in graph.children(node_name):
      if len(__QuantNodes) >= 1:
        break
      yield c.name

  breadth_first_search_handler(
      node.name, generator=__children_names, handler=__check_end)
  return len(__QuantNodes) == 0

def get_flows_and_info(quant_mode,
                       quantizer,
                       node_name=None,
                       params=None,
                       inputs=None):
  node = quantizer.configer.get_Nndctnode(node_name, params, inputs)
  return None, quantizer.configer.quant_input_names(
      node, inputs, params), (quantizer.configer.quant_output(node).name, True)

def process_inputs_and_params(node,
                              quantizer,
                              inputs,
                              params=[],
                              param_names=[]):

  # ignore parameters quantization if the node is not to be quantized
  #print('---- quant i/p: {}, in quant part: {}'.format(node.name, node.in_quant_part))
  if not node.in_quant_part or quantizer is None:
    return inputs, params

  # calculate quantization step of input activation
  # and quantize it
  quant_mode = quantizer.quant_mode
  if quantizer.need_quantize_tensor(node.name, 'input'):
    for idx in range(len(inputs)):
      if quant_mode in [1, 3]:
        inputs[idx] = quantizer.do_scan(
                        inputs[idx],
                        node.name,
                        node,
                        tensor_type='input')
      elif quant_mode == 2:
        inputs[idx] = quantizer.do_quantize(
            inputs[idx], node.name, node, tensor_type='input')

  #if len(params) > 0:
  #  print('---- quant p: {}'.format(node.name))
  # calculate quantization step of parameters
  # and quantize it
  if quant_mode in [1, 3]:
    for idx in range(len(params)):
      quantizer.do_scan(
          params[idx],
          param_names[idx],
          node,
          tensor_type='param')

  # only quantize parameters by pre-calculated step
  if quant_mode == 2:
    for idx in range(len(params)):
      params[idx] = quantizer.do_quantize(
          params[idx], param_names[idx], node, tensor_type='param')

  return inputs, params

def post_quant_process(node, outputs=[]):

  quant_mode, quantizer = maybe_get_quantizer()
  # ignore parameters quantization if the node is not to be quantized
  #print('---- quant o: {}, in quant part:{}'.format(node.name, node.in_quant_part))
  if ((not node.in_quant_part and 
       not node.op.is_custom_op) or 
      quantizer is None):
    return outputs

  if quantizer.need_quantize_tensor(node.name, 'output'):
    #print('---- quant o: {}'.format(node.name))
    output_name = node.name
    #print('qmode = %d, q_end: %d activation: %s' %
    #         (quant_mode, is_quant_end, output_name))
    if quant_mode in [1, 3]:
      for idx in range(len(outputs)):
        quantizer.do_scan(
            outputs[idx],
            output_name,
            node,
            tensor_type='output')
    elif quant_mode == 2:
      for idx in range(len(outputs)):
        outputs[idx] = quantizer.do_quantize(
            outputs[idx], output_name, node, tensor_type='output')

  return outputs

def quant_reluk_params(node, channel_max):
    
  quant_mode, quantizer = maybe_get_quantizer()
  # ignore parameters quantization if the node is not to be quantized
  #print('---- quant o: {}, in quant part:{}'.format(node.name, node.in_quant_part))
  if not node.in_quant_part or quantizer is None:
    return channel_max

  if quantizer.need_quantize_tensor(node.name, 'output'):
    #print('---- quant o: {}'.format(node.name))
    output_name = node.name
    #print('qmode = %d, q_end: %d activation: %s' %
    #         (quant_mode, is_quant_end, output_name))
    if quant_mode == 2:
      channel_max = quantizer.do_quantize(
        channel_max, output_name, node, tensor_type='output')

  return channel_max

def quant_channel_scale_params(node, channel_scale):
  quant_mode, quantizer = maybe_get_quantizer()
  # ignore parameters quantization if the node is not to be quantized
  #print('---- quant o: {}, in quant part:{}'.format(node.name, node.in_quant_part))
  if not node.in_quant_part or quantizer is None:
    return channel_scale

  if quantizer.need_quantize_tensor(node.name, 'output'):
    #print('---- quant o: {}'.format(node.name))
    output_name = node.name
    #print('qmode = %d, q_end: %d activation: %s' %
    #         (quant_mode, is_quant_end, output_name))
    if quant_mode == 2:
      channel_scale = quantizer.do_quantize(
        channel_scale, output_name, node, tensor_type='output')

  return channel_scale

class QuantizeData(object):
  def __init__(self, name, data):
    self._num_bins = 2048
    self._name = name
    self._data = data
    self._max = np.max(np.fabs(self._data))
    self._hist_interval = self._max / self._num_bins
    self._hist = self._load_data_into_bins(self._data.flatten()) 
    self._normalize_histogram()
    
  
  def kl_div(self, bn, fp):
    threshold_bin = int((bn / fp) / self._hist_interval) + 1
    threshold_hist = self._build_threshold_dist(threshold_bin)
    threshold_bin = threshold_hist.size
    target_bin = bn
    num_per_bin = float(threshold_bin) / target_bin
    quant_dist = self._build_quantize_dist(num_per_bin, target_bin)
    expand_q_dist = self._expand_quantize_dist(quant_dist, num_per_bin, threshold_bin)
    return self._compute_kl_div(threshold_hist, expand_q_dist)
    
   
  def _load_data_into_bins(self, data):
    abs_data = (np.fabs(data) / self._hist_interval).astype(np.int32)
    abs_data = np.where(abs_data < 2048, abs_data, 2047)                                 
    return np.bincount(abs_data)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

  
  def _normalize_histogram(self):
    self._hist = self._hist / self._hist.sum()
    
  def _build_threshold_dist(self, threshold_bin):
   
    if threshold_bin > self._hist.size:
      return self._hist.copy()
    
    threshold_sum = 0
    for i in range(threshold_bin, self._hist.size):
      threshold_sum += self._hist[i]
    t_dist = self._hist[:threshold_bin].copy()
    t_dist[threshold_bin - 1] += threshold_sum
    
    return t_dist
  
  def _build_quantize_dist(self, num_per_bin, target_bin):
    # num_per_bin = float(self._threshold_bin) / self._target_bin
    quant_dist = np.zeros(target_bin)
    for i in range(target_bin):
      start = i * num_per_bin
      end = start + num_per_bin
      
      left_upper = int(math.ceil(start))
      if left_upper > start:
        left_scale = float(left_upper) - start
        quant_dist[i] += left_scale * self._hist[left_upper - 1]
      
      right_lower = int(math.floor(end))
      if right_lower < end:
        right_scale = end - float(right_lower)
        quant_dist[i] += right_scale * self._hist[right_lower]
      
      quant_dist[i] += self._hist[left_upper:right_lower].sum()
      
    return quant_dist
     
  def _expand_quantize_dist(self, quant_dist, num_per_bin, threshold_bin):
    # num_per_bin = float(self._threshold_bin) / self._target_bin
    expand_q_dist = np.zeros(threshold_bin)
    for q_i in range(quant_dist.size):
      start = q_i * num_per_bin
      end = start + num_per_bin
      count = 0.0
      left_upper = int(math.ceil(start))
      if left_upper > start:
        left_scale = float(left_upper) - start
        if self._hist[left_upper - 1] != 0:
          count += left_scale
      
      right_lower = int(math.floor(end))
      if right_lower < end:
        right_scale = end - float(right_lower)
        if self._hist[right_lower] != 0:
          count += right_scale
      
      for i in range(left_upper, right_lower):
        if self._hist[i] != 0:
          count += 1
          
      expand_value = quant_dist[q_i] / count
          
      if left_upper > start:
        if self._hist[left_upper - 1] != 0:
          expand_q_dist[left_upper - 1] += expand_value * left_scale
      
      if right_lower < end:
        if self._hist[right_lower] != 0:
          expand_q_dist[right_lower] += expand_value * right_scale
          
      for i in range(left_upper, right_lower):
        if self._hist[i] != 0:
          expand_q_dist[i] += expand_value
          
    return expand_q_dist
  
  @staticmethod        
  def _compute_kl_div(t_dist, expand_q_dist):
    assert(t_dist.size == expand_q_dist.size)
    result = 0.0
    for i in range(t_dist.size):
      if t_dist[i] != 0:
        if expand_q_dist[i] == 0:
          result += 1
        else:
          result += t_dist[i] * math.log(t_dist[i] / expand_q_dist[i])
          
    return result
             
  def quant_efficiency(self, quant_data, bw):
    float_data = np.fabs(self._data.flatten())
    quant_data = np.fabs(quant_data.flatten())
    q_noise = np.square(float_data - quant_data).mean()
    sqnr = 10 * np.log10(np.square(float_data).mean() / q_noise)
    return sqnr / bw, sqnr
  
  def quant_mean_shift(self, quant_data):
    if quant_data.ndim != 4:
      float_data = self._data.flatten().mean()
      quant_data = quant_data.flatten().mean()
      return (quant_data - float_data) / float_data * 100, None
    else:
      float_data = np.mean(self._data, axis=(1, 2, 3))
      quant_data = np.mean(quant_data, axis=(1, 2, 3))
      shift = quant_data - float_data
      return (shift / float_data * 100).mean(), shift
    
  def all_close(self, quant_data):
    np.allclose(self._data, quant_data)
