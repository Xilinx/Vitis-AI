

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

from collections import namedtuple

import numpy as np

from nndct_shared.base import NNDCT_OP
from nndct_shared.nndct_graph import GraphSearcher
# from nndct_shared.nndct_graph.operator_definition import Conv2d
from nndct_shared.utils import NndctOption, PatternType
from .fuse_conv_bn import ConvBnHandler

#_ACT_TYPES = [NNDCT_OP.RELU, NNDCT_OP.RELU6]
_ACT_TYPES = [NNDCT_OP.RELU, NNDCT_OP.RELUK]
_POOL_TYPES = [NNDCT_OP.MAX_POOL, NNDCT_OP.ADAPTIVEAVGPOOL2D, NNDCT_OP.AVG_POOL]
_CLE_TYPES = [NNDCT_OP.CONV2D, NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.CONVTRANSPOSE2D]
CleInfo = namedtuple("CleInfo", ["conv_group", "scale_factor"])


class OptimizeCommander(object):
  def __init__(self, graph):
    self._graph = graph
    
  def FuseBnToConv(self):
    # find fusable bathnorm node
    fuse_bn_handler = ConvBnHandler()
    graph_searcher = GraphSearcher(self._graph)
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=[NNDCT_OP.CONV2D, 
                                                                          NNDCT_OP.BATCH_NORM],
                                                                 action=fuse_bn_handler), 
                                                     PatternType(pattern=[NNDCT_OP.DEPTHWISE_CONV2D, 
                                                                          NNDCT_OP.BATCH_NORM], 
                                                                 action=fuse_bn_handler),
                                                     PatternType(pattern=[NNDCT_OP.CONVTRANSPOSE2D, 
                                                                          NNDCT_OP.BATCH_NORM],
                                                                 action=fuse_bn_handler)])
    for id, node_list in node_sets.items():
      for nodeset in node_list:
        _, bn_node = nodeset
        self._graph.remove_node(bn_node)
  
  def equalize_weights_cross_conv_layers(self):
    r""" 
    This function re-implements the weight equalization technique proposed in the following paper.
    "Markus Nagel et al., Data-Free Quantization through Weight Equalization and Bias Correction",
    arXiv:1906.04721, 2019."
    """
    conv_layer_groups = []
    node = self._graph.get_node_by_idx(0)
    # build conv_layer_groups
    self._collect_layer_groups(node, conv_layer_groups)
    ''' 
    for id, group in enumerate(conv_layer_groups, 1):
      print(f"\ngroup{id}:")
      print("{")
      for n in group:
        print(f"  {n.name}")
      print("}")
    '''
      
    # generate equalized_group 
    equalized_groups = self._collect_equalized_groups(conv_layer_groups)
    '''
    for id, group in enumerate(equalized_groups, 1):
      print(f"\ngroup{id}:")
      print("{")
      for n in group:
        print(f"  {n.name}")
      print("}")
    '''
    # do equalization
    for i in range(1):
      cle_info_set = self._cross_layer_equalization_for_conv(equalized_groups)
    
  def _collect_layer_groups(self, node, conv_layer_groups, visited=None, conv_group=None):
  
    def _gather_conv_group(group): 
      if len(group) > 1 and group not in conv_layer_groups:
        conv_layer_groups.append(group)
      return []
    
    if not visited:
      visited = []
      
    if not conv_group:
      conv_group = []
    
    if node in visited:
      return 
    
    visited.append(node)
    
    if node.op.type in _CLE_TYPES:
      conv_group.append(node)
      if len(node.out_nodes) > 1:
        conv_group = _gather_conv_group(conv_group)    
    elif node.op.type in _ACT_TYPES:
      if node.op.type == NNDCT_OP.RELUK:
        conv_group.append(node)
      if len(node.out_nodes) > 1:
        conv_group = _gather_conv_group(conv_group)
    elif node.op.type in _POOL_TYPES:
      if len(node.out_nodes) > 1:
        conv_group = _gather_conv_group(conv_group)
    else:
      conv_group = _gather_conv_group(conv_group)
    
    for c_node in self._graph.children(node):
        self._collect_layer_groups(c_node, conv_layer_groups, visited, conv_group)
    
    _gather_conv_group(conv_group)    
  
  '''
  def _high_bias_fold(self, cle_info_set, fused_conv_bn_info):
    for cle_info in cle_info_set:
      conv_0 = cle_info.conv_group[0]
      conv_1 = cle_info.conv_group[1]
      conv_0_has_bias = conv_0.node_attr(conv_0.op.AttrName.BIAS_TERM)
      conv_1_has_bias = conv_1.node_attr(conv_1.op.AttrName.BIAS_TERM)
      if((conv_0_has_bias is False) or (conv_1_has_bias is False) or (conv_0 not in fused_conv_bn_info)):
          continue
      conv_0 = cle_info.conv_group[0]
      conv_1 = cle_info.conv_group[1]
      bn_params = fused_conv_bn_info[cle_info.conv_group[0]]
      bn_gamma = bn_params["gamma"] / cle_info.scale_factor
      bn_beta = bn_params["beta"] / cle_info.scale_factor
      conv_0_bias = conv_0.op.params[conv_0.op.ParamName.BIAS].data
      conv_1_bias = conv_1.op.params[conv_1.op.ParamName.BIAS].data
      conv_1_weight = conv_1.op.params[conv_1.op.ParamName.WEIGHTS].data
      if all([self._graph.node(node_name).op.type not in _ACT_TYPES for node_name in cle_info.conv_group[0].out_nodes]):
        absorb_bias = bn_beta.copy()
      else:
        absorb_bias = np.zeros_like(bn_beta)
        absorb_bias = np.where(bn_beta - 3 * np.fabs(bn_gamma) > 0, bn_beta - 3 * np.fabs(bn_gamma), absorb_bias)
      
      conv_0_bias -= absorb_bias
      conv_1_weight_reduced = np.sum(conv_1_weight, axis=(1, 2))
      if conv_1_weight.shape[0] == 1:
        bias_correction = conv_1_weight_reduced.squeeze(axis=0) * absorb_bias
      else:
        bias_correction = np.matmul(conv_1_weight_reduced, absorb_bias)
      conv_1_bias += bias_correction
      
      conv_0.op.params[conv_0.op.ParamName.BIAS].from_ndarray(conv_0_bias)
      conv_1.op.params[conv_1.op.ParamName.BIAS].from_ndarray(conv_1_bias)
  '''
  
  @staticmethod
  def _get_weight_data(node, layout):
    if node.op.type in [NNDCT_OP.CONV2D, NNDCT_OP.CONVTRANSPOSE2D]:
      if layout == "OHWI":
        return node.op.params[node.op.ParamName.WEIGHTS].data.copy()
      elif layout == "IHWO":
        return node.op.params[node.op.ParamName.WEIGHTS].data.copy().transpose(3, 1, 2, 0)
      else:
        raise ValueError("only support OHWI/IHWO layout for weight")
    elif node.op.type == NNDCT_OP.DEPTHWISE_CONV2D:
      # [channel_multipier, H, W, I]
      weight = node.op.params[node.op.ParamName.WEIGHTS].data.copy()
      # [I, H, W, channel_multiplier]
      weight = weight.transpose(3, 1, 2, 0)
      # depthwise always put channel num as first dimension
      return weight
    
  @staticmethod
  def _set_weight_data(node, data, layout):
    if node.op.type in [NNDCT_OP.CONV2D, NNDCT_OP.CONVTRANSPOSE2D]:
      if layout == "OHWI":
        node.op.params[node.op.ParamName.WEIGHTS].from_ndarray(data)
      elif layout == "IHWO":
        data = data.transpose(3, 1, 2, 0)
        node.op.params[node.op.ParamName.WEIGHTS].from_ndarray(data)
      else:
        raise ValueError("only support OHWI/IHWO layout for weight")
    elif node.op.type == NNDCT_OP.DEPTHWISE_CONV2D:
      data = data.transpose(3, 1, 2, 0)
      node.op.params[node.op.ParamName.WEIGHTS].from_ndarray(data)
  
  @staticmethod
  def _get_bias_data(node):
    return node.op.params[node.op.ParamName.BIAS].data.copy()
  
  @staticmethod
  def _set_bias_data(node, data):
    node.op.params[node.op.ParamName.BIAS].from_ndarray(data)
      
  @classmethod
  def _scale_weights_with_depthwise_conv(cls, group):
    # [O,H, W, I]
    conv_0_weight = cls._get_weight_data(group[0], layout="OHWI")
    conv_0_bias = None
    if group[0].node_attr(group[0].op.AttrName.BIAS_TERM):
      conv_0_bias = cls._get_bias_data(group[0])
    
    # [I, H, W, channel_multiplier]
    conv_1_weight = cls._get_weight_data(group[1], layout="IHWO")
   
    conv_1_bias = None
    if group[1].node_attr(group[1].op.AttrName.BIAS_TERM):
      conv_1_bias = cls._get_bias_data(group[1])
  
    # [O, H, W, I]
    conv_2_weight = cls._get_weight_data(group[2], layout="OHWI")
    # compute s01 and s12:
    # s01 = range0 / cuberoot(range0 * range1 * range2)
    # s12 = cuberoot(range0 * range1 * range2) / range2
    range_0 = np.max(np.fabs(conv_0_weight), axis=(1, 2, 3))
    range_1 = np.max(np.fabs(conv_1_weight), axis=(1, 2, 3))
    range_2 = np.max(np.fabs(conv_2_weight), axis=(0, 1, 2))
    cbrt_of_ranges = np.cbrt(range_0 * range_1 * range_2)
    scale_0_1 = np.ones_like(range_0)
    scale_1_2 = np.ones_like(range_1)
    
    scale_0_1 = np.where(cbrt_of_ranges != 0, range_0 / cbrt_of_ranges, scale_0_1)
    scale_1_2 = np.where(cbrt_of_ranges != 0, cbrt_of_ranges / range_2, scale_1_2)
    
    conv_0_weight = conv_0_weight.transpose(3, 1, 2, 0) / scale_0_1
    conv_0_weight = conv_0_weight.transpose(3, 1, 2, 0)
    if conv_0_bias is not None:
      conv_0_bias = conv_0_bias / scale_0_1
    
    # [channel_multiplier, H, W, I]
    conv_1_weight = conv_1_weight.transpose(3, 1, 2, 0)
    conv_1_weight = conv_1_weight * scale_0_1 / scale_1_2
    conv_1_weight = conv_1_weight.transpose(3, 1, 2, 0)
    if conv_1_bias is not None:
      conv_1_bias = conv_1_bias / scale_1_2
      
    conv_2_weight = conv_2_weight * scale_1_2
    
    cls._set_weight_data(group[0], conv_0_weight, layout="OHWI")
    if conv_0_bias is not None:
      cls._set_bias_data(group[0], conv_0_bias)
    
    cls._set_weight_data(group[1], conv_1_weight, layout="IHWO") 
    if conv_1_bias is not None:
      cls._set_bias_data(group[1], conv_1_bias)
    
    cls._set_weight_data(group[2], conv_2_weight, layout="OHWI")
    return scale_0_1, scale_1_2
  
  @classmethod
  def _scale_weights_and_bias_with_depthwise_conv(cls, group):
    #print('****************************_scale_weights_and_bias_with_**********depthwise_conv***************************')
    # [O,H, W, I]
    conv_0_weight = cls._get_weight_data(group[0], layout="OHWI")
    conv_0_bias = None
    if group[0].node_attr(group[0].op.AttrName.BIAS_TERM):
      conv_0_bias = cls._get_bias_data(group[0])
    
    # [I, H, W, channel_multiplier]
    conv_1_weight = cls._get_weight_data(group[1], layout="IHWO")
   
    conv_1_bias = None
    if group[1].node_attr(group[1].op.AttrName.BIAS_TERM):
      conv_1_bias = cls._get_bias_data(group[1])
  
    # [O, H, W, I]
    conv_2_weight = cls._get_weight_data(group[2], layout="OHWI")
    # compute s01 and s12:
    # s01 = range0 / cuberoot(range0 * range1 * range2)
    # s12 = cuberoot(range0 * range1 * range2) / range2
    # range_0 = np.max(np.fabs(conv_0_weight), axis=(1, 2, 3))
    conv_0_weight_ihw = conv_0_weight.reshape(conv_0_weight.shape[0], -1)
      
    if conv_0_bias is not None:
      conv_0_bias_clamp = conv_0_bias.copy().reshape(-1, 1)
      if np.count_nonzero(conv_0_weight_ihw) != conv_0_weight_ihw.size:
        #print(f"{group[0].name}' weight has zero element")
        conv_0_weight_ihw_clamp = conv_0_weight_ihw.copy()
        for channel in range(conv_0_weight_ihw_clamp.shape[0]):
          if np.count_nonzero(conv_0_weight_ihw_clamp[channel]) == 0:
            conv_0_bias_clamp[channel] = 0.0
            conv_0_weight_ihw_clamp[channel] = 1e-7
          elif np.count_nonzero(conv_0_weight_ihw_clamp[channel]) != conv_0_weight_ihw_clamp[channel].size:
            minval = np.min(np.fabs(np.ma.masked_where(conv_0_weight_ihw_clamp[channel] == 0,
                                    conv_0_weight_ihw_clamp[channel])))
            minval = 1e-7 if np.fabs(minval) < 1e-7 else minval
            conv_0_weight_ihw_clamp[channel] = np.where(np.fabs(conv_0_weight_ihw_clamp[channel]) < 1e-7, 
                                                        minval, conv_0_weight_ihw_clamp[channel])
        
        conv_0_weight_ihw_clamp = np.where(np.fabs(conv_0_weight_ihw_clamp) < 1e-7, 1e-7, conv_0_weight_ihw_clamp)
        factor = np.fabs(conv_0_bias_clamp) / np.fabs(conv_0_weight_ihw_clamp) 
      else:
        conv_0_weight_ihw_clamp = conv_0_weight_ihw.copy()
        conv_0_weight_ihw_clamp = np.where(np.fabs(conv_0_weight_ihw_clamp) < 1e-7, 1e-7, conv_0_weight_ihw_clamp)
        factor = np.fabs(conv_0_bias_clamp) / np.fabs(conv_0_weight_ihw_clamp) 
      '''  
      print(group[0].name, factor.mean())
      
      print('conv_0_weight_ihw_clamp, max: {}, min: {}, mean: {}, median: {}'.format(np.fabs(conv_0_weight_ihw_clamp).max(), 
                                                                                     np.fabs(conv_0_weight_ihw_clamp).min(),
                                                                                     np.fabs(conv_0_weight_ihw_clamp).mean(),
                                                                                     np.median(np.fabs(conv_0_weight_ihw_clamp))))
      print('conv_0_bias_clamp, max: {}, min: {}, mean: {}, median: {}'.format(np.fabs(conv_0_bias_clamp).max(), 
                                                                              np.fabs(conv_0_bias_clamp).min(),
                                                                              np.fabs(conv_0_bias_clamp).mean(),
                                                                              np.median(np.fabs(conv_0_bias_clamp))))
      print('factor, max: {}, min: {}, mean: {}, median: {}'.format(np.fabs(factor).max(), 
                                                                    np.fabs(factor).min(),
                                                                    np.fabs(factor).mean(),
                                                                    np.median(np.fabs(factor))))
      '''
      '''
      if np.fabs(conv_0_bias).max() < 10:
        if factor.mean() > 1000:
          shrink_factor = 5
        else:
          shrink_factor = 2
      else:
        if factor.mean() > 1000:
          shrink_factor = 100
        elif factor.mean() > 200:
          shrink_factor = 20
        elif factor.mean() > 100:
          shrink_factor = 10
        else:
          shrink_factor = 5
      '''
      
      if (np.fabs(conv_0_bias).max() < 10) and (np.fabs(conv_0_bias).max()/np.fabs(conv_0_weight_ihw_clamp).max() < 20):
        #if factor.mean() > 1000:
        if (np.median(factor) > 100) or (factor.mean() > 1000):
          shrink_factor = 5
        else:
          shrink_factor = 2
      else:
        #if factor.mean() > 1000:
        if np.median(factor) > 100 or factor.mean() > 1000:
          shrink_factor = 100
        #elif factor.mean() > 200:
        elif np.median(factor) > 30 or factor.mean() > 200:
          shrink_factor = 20
        #elif factor.mean() > 100:
        elif np.median(factor) > 15 or factor.mean() > 100:
          shrink_factor = 10
        else:
          shrink_factor = 5
      #shrink_factor = 2
      #print('**********************shrink_factor={}*****************************'.format(shrink_factor))
      conv_0_weight_bias = np.concatenate((conv_0_weight_ihw, conv_0_bias_clamp / shrink_factor), axis=1)
    else:
      conv_0_weight_bias = conv_0_weight_ihw   
    
    # range_0 = np.sqrt(conv_0_c_range * conv_0_bias_range)
    range_0 = np.max(np.fabs(conv_0_weight_bias), axis=1) 
    
    # range_1 = np.max(np.fabs(conv_1_weight), axis=(1, 2, 3))
    conv_1_weight_ihw = conv_1_weight.reshape(conv_1_weight.shape[0], -1)

    if conv_1_bias is not None:
      conv_1_bias_clamp = conv_1_bias.copy().reshape(-1, 1)
      if np.count_nonzero(conv_1_weight_ihw) != conv_1_weight_ihw.size:
        #print(f"{group[1].name}' weight has zero element")
        conv_1_weight_ihw_clamp = conv_1_weight_ihw.copy()
        for channel in range(conv_1_weight_ihw_clamp.shape[0]):
          if np.count_nonzero(conv_1_weight_ihw_clamp[channel]) == 0:
            conv_1_bias_clamp[channel] = 0.0
            conv_1_weight_ihw_clamp[channel] = 1e-7
          elif np.count_nonzero(conv_1_weight_ihw_clamp[channel]) != conv_1_weight_ihw_clamp[channel].size:
            minval = np.min(np.fabs(np.ma.masked_where(conv_1_weight_ihw_clamp[channel] == 0.0,
                                    conv_1_weight_ihw_clamp[channel])))
            minval = 1e-7 if np.fabs(minval) < 1e-7 else minval
            conv_1_weight_ihw_clamp[channel] = np.where(np.fabs(conv_1_weight_ihw_clamp[channel]) < 1e-7, 
                                                        minval, conv_1_weight_ihw_clamp[channel])
        
        conv_1_weight_ihw_clamp = np.where(np.fabs(conv_1_weight_ihw_clamp) < 1e-7, 1e-7, conv_1_weight_ihw_clamp)
        factor = np.fabs(conv_1_bias_clamp) / np.fabs(conv_1_weight_ihw_clamp) 
      else:
        conv_1_weight_ihw_clamp = conv_1_weight_ihw.copy()
        conv_1_weight_ihw_clamp = np.where(np.fabs(conv_1_weight_ihw_clamp) < 1e-7, 1e-7, conv_1_weight_ihw_clamp)
        factor = np.fabs(conv_1_bias_clamp) / np.fabs(conv_1_weight_ihw_clamp) 
        
        
      # shrink_factor = factor.mean() / magic_num
      '''
      print(group[1].name, factor.mean())
      
      print('conv_1_weight_ihw_clamp, max: {}, min: {}, mean: {}, median: {}'.format(np.fabs(conv_1_weight_ihw_clamp).max(), 
                                                                                    np.fabs(conv_1_weight_ihw_clamp).min(),
                                                                                    np.fabs(conv_1_weight_ihw_clamp).mean(),
                                                                                    np.median(np.fabs(conv_1_weight_ihw_clamp))))
      print('conv_1_bias_clamp, max: {}, min: {}, mean: {}, median: {}'.format(np.fabs(conv_1_bias_clamp).max(), 
                                                                              np.fabs(conv_1_bias_clamp).min(),
                                                                              np.fabs(conv_1_bias_clamp).mean(),
                                                                              np.median(np.fabs(conv_1_bias_clamp))))
      print('factor, max: {}, min: {}, mean: {}, median: {}'.format(np.fabs(factor).max(), 
                                                                    np.fabs(factor).min(),
                                                                    np.fabs(factor).mean(),
                                                                    np.median(np.fabs(factor))))
      '''
    
      '''
      if np.fabs(conv_1_bias).max() < 10:
        if factor.mean() > 1000:
          shrink_factor = 5
        else:
          shrink_factor = 2
      else:
        if factor.mean() > 1000:
          shrink_factor = 100
        elif factor.mean() > 200:
          shrink_factor = 20
        elif factor.mean() > 100:
          shrink_factor = 10
        else:
          shrink_factor = 5
      '''
      if (np.fabs(conv_1_bias).max() < 10) and (np.fabs(conv_0_bias).max()/np.fabs(conv_1_weight_ihw_clamp).max() < 20):
        #if factor.mean() > 1000:
        if np.median(factor) > 100 or factor.mean() > 1000:
          shrink_factor = 5
        else:
          shrink_factor = 2
      else:
        #if factor.mean() > 1000:
        if np.median(factor) > 100 or factor.mean() > 1000:
          shrink_factor = 100
        #elif factor.mean() > 200:
        elif np.median(factor) > 30 or factor.mean() > 200:
          shrink_factor = 20
        #elif factor.mean() > 100:
        elif np.median(factor) > 15 or factor.mean() > 100:
          shrink_factor = 10
        else:
          shrink_factor = 5
      #shrink_factor = 2
      #print('**********************shrink_factor={}*****************************'.format(shrink_factor))
      conv_1_weight_bias = np.concatenate((conv_1_weight_ihw, conv_1_bias_clamp / shrink_factor), axis=1)
    else:
      conv_1_weight_bias = conv_1_weight_ihw   
    
    range_1 = np.max(np.fabs(conv_1_weight_bias), axis=1)
                 
    range_2 = np.max(np.fabs(conv_2_weight), axis=(0, 1, 2))
    cbrt_of_ranges = np.cbrt(range_0 * range_1 * range_2)
    scale_0_1 = np.ones_like(range_0)
    scale_1_2 = np.ones_like(range_1)
    
    scale_0_1 = np.where(cbrt_of_ranges != 0, range_0 / cbrt_of_ranges, scale_0_1)
    scale_1_2 = np.where(cbrt_of_ranges != 0, cbrt_of_ranges / range_2, scale_1_2)
    
    scale_0_1 = np.where((range_0 + range_1) < 0.5, 1, scale_0_1)
    scale_1_2 = np.where((range_2 + range_1) < 0.5, 1, scale_1_2)
    
    conv_0_weight = conv_0_weight.transpose(3, 1, 2, 0) / scale_0_1
    conv_0_weight = conv_0_weight.transpose(3, 1, 2, 0)
    if conv_0_bias is not None:
      conv_0_bias = conv_0_bias / scale_0_1
    
    # [channel_multiplier, H, W, I]
    conv_1_weight = conv_1_weight.transpose(3, 1, 2, 0)
    conv_1_weight = conv_1_weight * scale_0_1 / scale_1_2
    conv_1_weight = conv_1_weight.transpose(3, 1, 2, 0)
    if conv_1_bias is not None:
      conv_1_bias = conv_1_bias / scale_1_2
      
    conv_2_weight = conv_2_weight * scale_1_2
    
    cls._set_weight_data(group[0], conv_0_weight, layout="OHWI")
    if conv_0_bias is not None:
      cls._set_bias_data(group[0], conv_0_bias)
    
    cls._set_weight_data(group[1], conv_1_weight, layout="IHWO") 
    if conv_1_bias is not None:
      cls._set_bias_data(group[1], conv_1_bias)
    
    cls._set_weight_data(group[2], conv_2_weight, layout="OHWI")
    return scale_0_1, scale_1_2
  
  @classmethod
  def _scale_weights_and_bias_with_conv(cls, group):
    #print('-------------------_scale_weights_and_bias_with_conv---------------------')
    # conv: [O,H, W, I] / depthwise_conv: [I, H, W, channel_multiplier]
    conv_0_weight = cls._get_weight_data(group[0], layout="OHWI")
    conv_0_bias = None
    if group[0].node_attr(group[0].op.AttrName.BIAS_TERM):
      conv_0_bias = cls._get_bias_data(group[0])
  
    
    # conv: [O, H, W, I] / depthwise_conv: [I, H, W, channel_multiplier]
    conv_1_weight = cls._get_weight_data(group[1], layout="OHWI")
    if group[1].op.type == NNDCT_OP.DEPTHWISE_CONV2D:
      # depthwise_conv:  [I, H, W, channel_multiplier] -> [channel_muliplier, H, W, I]
      conv_1_weight = conv_1_weight.transpose(3, 1, 2, 0)
    # compute scale:
    # scale = range0 / sqrt(range0 * range1)
    conv_0_weight_ihw = conv_0_weight.reshape(conv_0_weight.shape[0], -1)
   
        
    if conv_0_bias is not None: 
      conv_0_bias_clamp = conv_0_bias.copy().reshape(-1, 1)
      if np.count_nonzero(conv_0_weight_ihw) != conv_0_weight_ihw.size:
        #print(f"{group[0].name}' weight has zero element")
        conv_0_weight_ihw_clamp = conv_0_weight_ihw.copy()
        for channel in range(conv_0_weight_ihw_clamp.shape[0]):
          if np.count_nonzero(conv_0_weight_ihw_clamp[channel]) == 0:
            conv_0_bias_clamp[channel] = 0.0
            conv_0_weight_ihw_clamp[channel] = 1e-7
          elif np.count_nonzero(conv_0_weight_ihw_clamp[channel]) != conv_0_weight_ihw_clamp[channel].size:
            minval = np.min(np.fabs(np.ma.masked_where(conv_0_weight_ihw_clamp[channel] == 0.0,
                                    conv_0_weight_ihw_clamp[channel])))
            conv_0_weight_ihw_clamp[channel] = np.where(conv_0_weight_ihw_clamp[channel] == 0.0, 
                                                        minval, conv_0_weight_ihw_clamp[channel])
          
        conv_0_weight_ihw_clamp = np.where(np.fabs(conv_0_weight_ihw_clamp) < 1e-7, 1e-7, conv_0_weight_ihw_clamp)
        factor = np.fabs(conv_0_bias_clamp) / np.fabs(conv_0_weight_ihw_clamp) 
      else:
        conv_0_weight_ihw_clamp = conv_0_weight_ihw.copy()
        conv_0_weight_ihw_clamp = np.where(np.fabs(conv_0_weight_ihw_clamp) < 1e-7, 1e-7, conv_0_weight_ihw_clamp)
        factor = np.fabs(conv_0_bias_clamp) / np.fabs(conv_0_weight_ihw_clamp)
      '''
      print(group[0].name, factor.mean())
      
      print('conv_0_weight_ihw_clamp, max: {}, min: {}, mean: {}, median: {}'.format(np.fabs(conv_0_weight_ihw_clamp).max(), 
                                                                                    np.fabs(conv_0_weight_ihw_clamp).min(),
                                                                                    np.fabs(conv_0_weight_ihw_clamp).mean(),
                                                                                    np.median(np.fabs(conv_0_weight_ihw_clamp))))
      print('conv_0_bias_clamp, max: {}, min: {}, mean: {}, median: {}'.format(np.fabs(conv_0_bias_clamp).max(), 
                                                                              np.fabs(conv_0_bias_clamp).min(),
                                                                              np.fabs(conv_0_bias_clamp).mean(),
                                                                              np.median(np.fabs(conv_0_bias_clamp))))
      print('factor, max: {}, min: {}, mean: {}, median: {}'.format(np.fabs(factor).max(), 
                                                                    np.fabs(factor).min(),
                                                                    np.fabs(factor).mean(),
                                                                    np.median(np.fabs(factor))))
      '''
      '''
      if np.fabs(conv_0_bias).max() < 10:
        if factor.mean() > 1000:
          shrink_factor = 5
        else:
          shrink_factor = 2
      else:
        if factor.mean() > 500:
          shrink_factor = 20
        elif factor.mean() > 100:
          shrink_factor = 10
        else:
          shrink_factor = 5
      '''
      if (np.fabs(conv_0_bias).max() < 10) and (np.fabs(conv_0_bias).max()/np.fabs(conv_0_weight_ihw_clamp).max() < 20):
        if np.median(factor) > 100 or factor.mean() > 1000:
          shrink_factor = 5
        else:
          shrink_factor = 2
      else:
        #if factor.mean() > 500:
        if np.median(factor) > 30 or factor.mean() > 500:
          shrink_factor = 20
        #elif factor.mean() > 100:
        elif np.median(factor) > 15 or factor.mean() > 100:
          shrink_factor = 10
        else:
          shrink_factor = 5
      #shrink_factor = 2
      #print('**********************shrink_factor={}*****************************'.format(shrink_factor))
      conv_0_weight_bias = np.concatenate((conv_0_weight_ihw, conv_0_bias_clamp / shrink_factor), axis=1)
    else:
      conv_0_weight_bias = conv_0_weight_ihw
                 
    range_0 = np.max(np.fabs(conv_0_weight_bias), axis=1)
    range_1 = np.max(np.fabs(conv_1_weight), axis=(0, 1, 2))
    sqrt_of_ranges = np.sqrt(range_0 * range_1)
    scale = np.ones_like(range_0)
    
    scale = np.where(sqrt_of_ranges != 0, range_0 / sqrt_of_ranges, scale)
    
    i_max = np.max(np.fabs(conv_0_weight_bias), axis=1)
    o_max = np.max(np.fabs(conv_1_weight), axis=(0, 1, 2))
    scale = np.where((i_max + o_max) < 0.5, 1, scale)
    
    conv_0_weight = conv_0_weight.transpose(3, 1, 2, 0) / scale
    conv_0_weight = conv_0_weight.transpose(3, 1, 2, 0)
    if conv_0_bias is not None:
      conv_0_bias = conv_0_bias / scale
      
    conv_1_weight = conv_1_weight * scale
    if group[1].op.type == NNDCT_OP.DEPTHWISE_CONV2D:
      # depthwise_conv: [channel_muliplier, H, W, I] -> [I, H, W, channel_multiplier]
      conv_1_weight = conv_1_weight.transpose(3, 1, 2, 0) 
    
    cls._set_weight_data(group[0], conv_0_weight, layout="OHWI")
    if conv_0_bias is not None:
      cls._set_bias_data(group[0], conv_0_bias)
    
    cls._set_weight_data(group[1], conv_1_weight, layout="OHWI")
    return scale
  
  @classmethod
  def _scale_weights_with_conv(cls, group):
    # conv: [O,H, W, I] / depthwise_conv: [I, H, W, channel_multiplier]
    conv_0_weight = cls._get_weight_data(group[0], layout="OHWI")
    conv_0_bias = None
    if group[0].node_attr(group[0].op.AttrName.BIAS_TERM):
      conv_0_bias = cls._get_bias_data(group[0])
    
    # conv: [O, H, W, I] / depthwise_conv: [I, H, W, channel_multiplier]
    conv_1_weight = cls._get_weight_data(group[1], layout="OHWI")
    if group[1].op.type == NNDCT_OP.DEPTHWISE_CONV2D:
      # depthwise_conv:  [I, H, W, channel_multiplier] -> [channel_muliplier, H, W, I]
      conv_1_weight = conv_1_weight.transpose(3, 1, 2, 0)
    # compute scale:
    # scale = range0 / sqrt(range0 * range1)
    range_0 = np.max(np.fabs(conv_0_weight), axis=(1, 2, 3))
    range_1 = np.max(np.fabs(conv_1_weight), axis=(0, 1, 2))
    sqrt_of_ranges = np.sqrt(range_0 * range_1)
    scale = np.ones_like(range_0)
    
    scale = np.where(sqrt_of_ranges != 0, range_0 / sqrt_of_ranges, scale)
    
    conv_0_weight = conv_0_weight.transpose(3, 1, 2, 0) / scale
    conv_0_weight = conv_0_weight.transpose(3, 1, 2, 0)
    if conv_0_bias is not None:
      conv_0_bias = conv_0_bias / scale
      
    conv_1_weight = conv_1_weight * scale
    if group[1].op.type == NNDCT_OP.DEPTHWISE_CONV2D:
      # depthwise_conv: [channel_muliplier, H, W, I] -> [I, H, W, channel_multiplier]
      conv_1_weight = conv_1_weight.transpose(3, 1, 2, 0) 
    
    cls._set_weight_data(group[0], conv_0_weight, layout="OHWI")
    if conv_0_bias is not None:
      cls._set_bias_data(group[0], conv_0_bias)
    
    cls._set_weight_data(group[1], conv_1_weight, layout="OHWI")
    return scale

  @classmethod
  def _scale_weights_and_bias_with_conv_reluk(cls, group):
    # conv: [O,H, W, I] / depthwise_conv: [I, H, W, channel_multiplier]
    conv_0_weight = cls._get_weight_data(group[0], layout="OHWI")
    conv_0_bias = None
    if group[0].node_attr(group[0].op.AttrName.BIAS_TERM):
      conv_0_bias = cls._get_bias_data(group[0])
    
    # conv: [O, H, W, I] / depthwise_conv: [I, H, W, channel_multiplier]
    conv_1_weight = cls._get_weight_data(group[-1], layout="OHWI")
    if group[-1].op.type == NNDCT_OP.DEPTHWISE_CONV2D:
      # depthwise_conv:  [I, H, W, channel_multiplier] -> [channel_muliplier, H, W, I]
      conv_1_weight = conv_1_weight.transpose(3, 1, 2, 0)
    # compute scale:
    conv_0_weight_ihw = conv_0_weight.reshape(conv_0_weight.shape[0], -1)
   
        
    if conv_0_bias is not None: 
      conv_0_bias_clamp = conv_0_bias.copy().reshape(-1, 1)
      if np.count_nonzero(conv_0_weight_ihw) != conv_0_weight_ihw.size:
        #print(f"{group[0].name}' weight has zero element")
        conv_0_weight_ihw_clamp = conv_0_weight_ihw.copy()
        for channel in range(conv_0_weight_ihw_clamp.shape[0]):
          if np.count_nonzero(conv_0_weight_ihw_clamp[channel]) == 0:
            conv_0_bias_clamp[channel] = 0.0
            conv_0_weight_ihw_clamp[channel] = 1e-7
          elif np.count_nonzero(conv_0_weight_ihw_clamp[channel]) != conv_0_weight_ihw_clamp[channel].size:
            minval = np.min(np.fabs(np.ma.masked_where(conv_0_weight_ihw_clamp[channel] == 0.0,
                                    conv_0_weight_ihw_clamp[channel])))
            conv_0_weight_ihw_clamp[channel] = np.where(conv_0_weight_ihw_clamp[channel] == 0.0, 
                                                        minval, conv_0_weight_ihw_clamp[channel])
          
        conv_0_weight_ihw_clamp = np.where(np.fabs(conv_0_weight_ihw_clamp) < 1e-7, 1e-7, conv_0_weight_ihw_clamp)
        factor = np.fabs(conv_0_bias_clamp) / np.fabs(conv_0_weight_ihw_clamp) 
      else:
        conv_0_weight_ihw_clamp = conv_0_weight_ihw.copy()
        conv_0_weight_ihw_clamp = np.where(np.fabs(conv_0_weight_ihw_clamp) < 1e-7, 1e-7, conv_0_weight_ihw_clamp)
        factor = np.fabs(conv_0_bias_clamp) / np.fabs(conv_0_weight_ihw_clamp)
      
      if (np.fabs(conv_0_bias).max() < 10) and (np.fabs(conv_0_bias).max()/np.fabs(conv_0_weight_ihw_clamp).max() < 20):
        if np.median(factor) > 100 or factor.mean() > 1000:
          shrink_factor = 5
        else:
          shrink_factor = 2
      else:
        #if factor.mean() > 500:
        if np.median(factor) > 30 or factor.mean() > 500:
          shrink_factor = 20
        #elif factor.mean() > 100:
        elif np.median(factor) > 15 or factor.mean() > 100:
          shrink_factor = 10
        else:
          shrink_factor = 5
      #shrink_factor = 2
      #print('**********************shrink_factor={}*****************************'.format(shrink_factor))
      conv_0_weight_bias = np.concatenate((conv_0_weight_ihw, conv_0_bias_clamp / shrink_factor), axis=1)
    else:
      conv_0_weight_bias = conv_0_weight_ihw
                 
    range_0 = np.max(np.fabs(conv_0_weight_bias), axis=1)
    range_1 = np.max(np.fabs(conv_1_weight), axis=(0, 1, 2))
    sqrt_of_ranges = np.sqrt(range_0 * range_1)
    scale = np.ones_like(range_0)
    
    scale = np.where(sqrt_of_ranges != 0, range_0 / sqrt_of_ranges, scale)
    
    i_max = np.max(np.fabs(conv_0_weight_bias), axis=1)
    o_max = np.max(np.fabs(conv_1_weight), axis=(0, 1, 2))
    scale = np.where((i_max + o_max) < 0.5, 1, scale)
    
    conv_0_weight = conv_0_weight.transpose(3, 1, 2, 0) / scale
    conv_0_weight = conv_0_weight.transpose(3, 1, 2, 0)
    if conv_0_bias is not None:
      conv_0_bias = conv_0_bias / scale
      
    conv_1_weight = conv_1_weight * scale
    if group[-1].op.type == NNDCT_OP.DEPTHWISE_CONV2D:
      # depthwise_conv: [channel_muliplier, H, W, I] -> [I, H, W, channel_multiplier]
      conv_1_weight = conv_1_weight.transpose(3, 1, 2, 0) 
    
    cls._set_weight_data(group[0], conv_0_weight, layout="OHWI")
    if conv_0_bias is not None:
      cls._set_bias_data(group[0], conv_0_bias)
    
    cls._set_weight_data(group[-1], conv_1_weight, layout="OHWI")
    
    if (len(group)==3) and (group[1].op.type=='reluk'):
      channel_num = conv_0_weight.shape[0]
      channel_max = group[1].op.channel_max
      if (isinstance(channel_max, float) 
        or isinstance(channel_max, int) 
        or (isinstance(channel_max, np.array) and channel_max.ndim == 1)):
        group[1].op.channel_max = (channel_max/scale).reshape((1,channel_num,1,1)).tolist()
      elif (isinstance(channel_max, np.array) and channel_max.ndim == 4):
        group[1].op.channel_max = (channel_max/(scale.reshape((1,channel_num,1,1)))).tolist()
    
    return scale
    
  @classmethod
  def _cross_layer_equalization_for_conv(cls, equalized_groups):
    iters = 1
    for i in range(iters):
      cle_info_set = []
      for group in equalized_groups:
        #print(group)
        '''
        if len(group) == 3:
          scale_factor = cls._scale_weights_and_bias_with_depthwise_conv(group)
        else:
          scale_factor = cls._scale_weights_and_bias_with_conv(group)
        '''
        scale_factor = cls._scale_weights_and_bias_with_conv_reluk(group)
    return cle_info_set
           
  @classmethod
  def _collect_equalized_groups(cls, layer_groups):
    equalized_groups = []
    for group in layer_groups:
      equalized_groups += cls._convert_layer_group_to_equalized_groups(group)

    return equalized_groups

  '''
  @staticmethod
  def _convert_layer_group_to_equalized_groups(layer_group):
    equalized_groups = []
    prev_conv, *left_convs = layer_group
    while left_convs:
      next_conv, *others = left_convs
      
      if next_conv.op.type == NNDCT_OP.DEPTHWISE_CONV2D:
        if others:
          next_non_depthwise_conv = others.pop(0)
          equalized_groups.append((prev_conv, next_conv, next_non_depthwise_conv))
          prev_conv = next_non_depthwise_conv
      else:
        equalized_groups.append((prev_conv, next_conv))
        prev_conv = next_conv
      
      #equalized_groups.append((prev_conv, next_conv))
      #prev_conv = next_conv
      
      left_convs = others
    return equalized_groups
    '''         

  @staticmethod
  def _convert_layer_group_to_equalized_groups(layer_group):
    equalized_groups = []
    prev_conv, *left_nodes = layer_group
    while left_nodes:
      next_node, *others = left_nodes
      if prev_conv.op.type in _CLE_TYPES:
        if next_node.op.type == NNDCT_OP.RELUK:
          if others:
            next_conv = others.pop(0)
            if next_conv.op.type in _CLE_TYPES:
              equalized_groups.append((prev_conv, next_node, next_conv))
            prev_conv = next_conv
        elif next_node.op.type in _CLE_TYPES:
          equalized_groups.append((prev_conv, next_node))
          prev_conv = next_node
        else:
          prev_conv = next_node
      else:
        prev_conv = next_node
      left_nodes = others
    return equalized_groups      
