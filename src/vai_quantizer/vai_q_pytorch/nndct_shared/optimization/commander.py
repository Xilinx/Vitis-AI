

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
import copy
import numpy as np
import math

from nndct_shared.base import NNDCT_OP
from nndct_shared.nndct_graph import Graph, Node, Tensor, GraphSearcher, graph_searcher
# from nndct_shared.nndct_graph.operator_definition import Conv2d
from nndct_shared.utils import NndctOption, PatternType, NndctScreenLogger, QError, QWarning, QNote
from .fuse_conv_bn import ConvBnHandler
from .fuse_embed_ln_actv import EmbedLnActvHandler
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.utils import  NndctOption
#from nndct_shared.optimization.parse_utils import _GRAPH_SCOPE_SYM, get_full_name

#_ACT_TYPES = [NNDCT_OP.RELU, NNDCT_OP.RELU6]
_ACT_TYPES = [NNDCT_OP.RELU, NNDCT_OP.RELUK]
_POOL_TYPES = [NNDCT_OP.MAX_POOL, NNDCT_OP.ADAPTIVEAVGPOOL2D, NNDCT_OP.AVG_POOL]
_CLE_TYPES = [NNDCT_OP.CONV2D, NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.CONVTRANSPOSE2D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D,\
              NNDCT_OP.CONV3D, NNDCT_OP.DEPTHWISE_CONV3D, NNDCT_OP.CONVTRANSPOSE3D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D]
_CONV_TYPES = [NNDCT_OP.CONV2D, NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.CONVTRANSPOSE2D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D,\
              NNDCT_OP.CONV3D, NNDCT_OP.DEPTHWISE_CONV3D, NNDCT_OP.CONVTRANSPOSE3D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D, \
              NNDCT_OP.CONV1D, NNDCT_OP.DEPTHWISE_CONV1D]
_BN_TYPES = [NNDCT_OP.BATCH_NORM]
#_CONV3D_TYPES = [NNDCT_OP.CONV3D, NNDCT_OP.CONVTRANSPOSE3D]

_OP_LAYOUTS = {NNDCT_OP.CONV2D: "OHWI", NNDCT_OP.CONVTRANSPOSE2D: "OHWI", \
               NNDCT_OP.CONV3D: "OHDWI", NNDCT_OP.CONVTRANSPOSE3D: "OHDWI"}
CleInfo = namedtuple("CleInfo", ["conv_group", "scale_factor"])

class OptimizeCommander(object):
  def __init__(self, graph):
    self._graph = graph
    self.graph_depth = graph.get_graph_depth()
    if NndctOption.nndct_traversal_graph_mode.value == 0:
      if self.graph_depth > 900:
        self._collect_layer_groups = self._collect_layer_groups_iteration
      else:
        self._collect_layer_groups = self._collect_layer_groups_recursion
    if NndctOption.nndct_traversal_graph_mode.value == 1:
        self._collect_layer_groups = self._collect_layer_groups_recursion
    if NndctOption.nndct_traversal_graph_mode.value == 2:
        self._collect_layer_groups = self._collect_layer_groups_iteration
    
  def SetNegativeSlope(self):
    for node in self._graph.nodes:
      if node.op.type == NNDCT_OP.LEAKY_RELU:
        quant_mode, _ = maybe_get_quantizer()
        if quant_mode is None or NndctOption.nndct_quant_off.value:
          NndctScreenLogger().warning2user_once(QWarning.LEAKYRELU, f"Preserve negative_slope({node.node_attr(node.op.AttrName.ALPHA)}) of LeakyReLU without quantization.")
        else:
          if node.node_attr(node.op.AttrName.ALPHA) != 0.1015625:
            NndctScreenLogger().warning2user_once(QWarning.LEAKYRELU, f"Force to change negative_slope of LeakyReLU from {node.node_attr(node.op.AttrName.ALPHA)} to 0.1015625 because DPU only supports this value. It is recommended to change all negative_slope of LeakyReLU to 0.1015625 and re-train the float model for better deployed model accuracy.")
          node.set_node_attr(node.op.AttrName.ALPHA, 0.1015625)

  def ConvertBNParams(self):
    for node in self._graph.nodes:
      if node.op.type in _BN_TYPES:
        NndctScreenLogger().info2user(QNote.NOT_FUSED_BN, f"Node {node.name} cannot be fused into CONV layers, this is not quantization friendly. It is recommended to adjsut the pattern to CONV+BN.")
        gamma = node.op.params[node.op.ParamName.GAMMA].data
        beta = node.op.params[node.op.ParamName.BETA].data
        mean = node.op.params[node.op.ParamName.MOVING_MEAN].data
        var = node.op.params[node.op.ParamName.MOVING_VAR].data
        epsilon = node.node_attr(node.op.AttrName.EPSILON)
        scale = gamma / np.sqrt(var + epsilon)
        offset = beta - mean * scale
        new_mean = mean.copy()
        new_var = var.copy()
        new_mean.fill(0)
        new_var.fill(1)
        node.set_node_attr(node.op.AttrName.EPSILON, 0.0)

        node.op.set_param_from_data(node.op.ParamName.GAMMA, scale)
        node.op.set_param_from_data(node.op.ParamName.BETA, offset)
        node.op.set_param_from_data(node.op.ParamName.MOVING_MEAN, new_mean)
        node.op.set_param_from_data(node.op.ParamName.MOVING_VAR, new_var)

  def FuseEmbedLnActv(self):
    fuse_embed_handler = EmbedLnActvHandler()
    graph_searcher = GraphSearcher(self._graph)
    node_sets = graph_searcher.find_nodes_from_type(
      [PatternType(pattern=[NNDCT_OP.EMBEDDING, NNDCT_OP.LAYER_NORM, NNDCT_OP.SIGMOID],
                   action=fuse_embed_handler),
       PatternType(pattern=[NNDCT_OP.EMBEDDING, NNDCT_OP.LAYER_NORM, NNDCT_OP.TANH],
                   action=fuse_embed_handler)])
    removed_nodes = set()
    for id, node_list in node_sets.items():
      for nodeset in node_list:
        ln_node = nodeset[1]
        if ln_node.merged and ln_node not in removed_nodes:
          self._graph.remove_node(ln_node)
          removed_nodes.add(ln_node)
        
        actv_node = nodeset[2]
        if actv_node.merged and actv_node not in removed_nodes:
          self._graph.remove_node(actv_node)
          removed_nodes.add(actv_node)
    
  def FuseBnToConv(self):
    # find fusable bathnorm node
    fuse_bn_handler = ConvBnHandler()
    graph_searcher = GraphSearcher(self._graph)
    node_sets = graph_searcher.find_nodes_from_type(
        [PatternType(pattern=[NNDCT_OP.CONV2D, NNDCT_OP.BATCH_NORM],
                     action=fuse_bn_handler), 
         PatternType(pattern=[NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.BATCH_NORM], 
                     action=fuse_bn_handler),
         PatternType(pattern=[NNDCT_OP.CONVTRANSPOSE2D, NNDCT_OP.BATCH_NORM],
                     action=fuse_bn_handler),
         PatternType(pattern=[NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D, NNDCT_OP.BATCH_NORM],
                     action=fuse_bn_handler),
         PatternType(pattern=[NNDCT_OP.CONV3D, NNDCT_OP.BATCH_NORM],
                     action=fuse_bn_handler),
         PatternType(pattern=[NNDCT_OP.DEPTHWISE_CONV3D, NNDCT_OP.BATCH_NORM],
                     action=fuse_bn_handler),
         PatternType(pattern=[NNDCT_OP.CONVTRANSPOSE3D, NNDCT_OP.BATCH_NORM],
                     action=fuse_bn_handler),
         PatternType(pattern=[NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D, NNDCT_OP.BATCH_NORM],
                     action=fuse_bn_handler),
         PatternType(pattern=[NNDCT_OP.CONV2D, NNDCT_OP.CONCAT, NNDCT_OP.BATCH_NORM],
                     action=fuse_bn_handler), 
         PatternType(pattern=[NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.CONCAT, NNDCT_OP.BATCH_NORM], 
                     action=fuse_bn_handler),
         PatternType(pattern=[NNDCT_OP.CONVTRANSPOSE2D, NNDCT_OP.CONCAT, NNDCT_OP.BATCH_NORM],
                     action=fuse_bn_handler),
         PatternType(pattern=[NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D, NNDCT_OP.CONCAT, NNDCT_OP.BATCH_NORM],
                     action=fuse_bn_handler),
         PatternType(pattern=[NNDCT_OP.CONV3D, NNDCT_OP.CONCAT, NNDCT_OP.BATCH_NORM],
                     action=fuse_bn_handler),
         PatternType(pattern=[NNDCT_OP.DEPTHWISE_CONV3D, NNDCT_OP.CONCAT, NNDCT_OP.BATCH_NORM],
                     action=fuse_bn_handler),
         PatternType(pattern=[NNDCT_OP.CONVTRANSPOSE3D, NNDCT_OP.CONCAT, NNDCT_OP.BATCH_NORM],
                     action=fuse_bn_handler),
         PatternType(pattern=[NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D, NNDCT_OP.CONCAT, NNDCT_OP.BATCH_NORM],
                     action=fuse_bn_handler),
        ])
    removed_bn = set()
    for id, node_list in node_sets.items():
      for nodeset in node_list:
        bn_node = nodeset[-1]
        if bn_node.merged and bn_node not in removed_bn:
          self._graph.remove_node(bn_node)
          removed_bn.add(bn_node)


  def DecoupleSharedParamsInConv(self):
    # decouple shared parameters in graph
    bias_tensor_list = []
    weight_tensor_list = []
    for node in self._graph.nodes:
      if node.op.type in _CONV_TYPES:
        weight_tensor = node.op.params[node.op.ParamName.WEIGHTS]
        weight_name = weight_tensor.name
        if weight_name in weight_tensor_list:
          node_idx = node.idx
          weight_name_copy = weight_name + '.' + str(node_idx)
          new_weight_tensor = Tensor(name=weight_name_copy)
          new_weight_tensor.clone_from(weight_tensor)
          node.op.set_param(node.op.ParamName.WEIGHTS, new_weight_tensor)
        else:
          weight_tensor_list.append(weight_name)         

        if node.node_attr(node.op.AttrName.BIAS_TERM):
          bias_tensor = node.op.params[node.op.ParamName.BIAS]
          bias_name = bias_tensor.name
          if bias_name in bias_tensor_list:
            node_idx = node.idx
            bias_name_copy = bias_name + '.' + str(node_idx)
            new_bias_tensor = Tensor(name=bias_name_copy)
            new_bias_tensor.clone_from(bias_tensor)
            node.op.set_param(node.op.ParamName.BIAS, new_bias_tensor)
          else:
            bias_tensor_list.append(bias_name)        
  
  def equalize_weights_cross_conv_layers(self):
    r""" 
    This function re-implements the weight equalization technique proposed in the following paper.
    "Markus Nagel et al., Data-Free Quantization through Weight Equalization and Bias Correction",
    arXiv:1906.04721, 2019."
    """
    conv_layer_groups = []
    # node = self._graph.get_node_by_idx(0)
    #node = list(self._graph.nodes)[0]
    input_nodes = self._graph.get_input_nodes()
    # build conv_layer_groups
    for node in input_nodes:
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
      
    return self._graph
                
  def _collect_layer_groups_iteration(self, node_in, conv_layer_groups_in, visited_in=None, conv_group_in=None):
    def _gather_conv_group(group): 
      if len(group) > 1 and group not in conv_layer_groups:
        conv_layer_groups.append(group)
      return []

    task_stack = [[node_in, conv_layer_groups_in, visited_in, conv_group_in, 1]]
    while len(task_stack) > 0:

        node, conv_layer_groups, visited, conv_group ,last_run_child_node_flag = task_stack.pop()
        if not visited:
          visited = []

        if not conv_group:
          conv_group = []

        if node in visited:
          last_run_child_node_flag = last_run_child_node_flag - 1
          if last_run_child_node_flag > 0:
            for k in range(last_run_child_node_flag):
                _gather_conv_group(conv_group)
          continue 

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

        new_task_list = []

        for c_node in self._graph.children(node):
            new_task_list.append([c_node, conv_layer_groups, visited, conv_group, 1])
        new_task_list.reverse()
        
        if len(new_task_list) > 0:
            new_task_list[0][-1] = last_run_child_node_flag +  new_task_list[0][-1]
            
        if len(new_task_list) == 0:
            for k in range(last_run_child_node_flag):
                _gather_conv_group(conv_group)
            
        for task in new_task_list:
            task_stack.append(task)
            
 

        
  def _collect_layer_groups_recursion(self, node, conv_layer_groups, visited=None, conv_group=None):
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
  def _compute_transpose_order(layout_src, layout_dest):
    assert len(layout_src) == len(layout_dest)
    transpose_order = []
    for dim in layout_dest:
      dim_order = layout_src.find(dim)
      assert dim_order >= 0
      transpose_order.append(dim_order)
    return tuple(transpose_order)
  
  @staticmethod
  def _get_weight_data(node, layout):
    if node.op.type in [NNDCT_OP.CONV2D, NNDCT_OP.CONVTRANSPOSE2D, NNDCT_OP.CONV3D, NNDCT_OP.CONVTRANSPOSE3D]:
      src_layout = _OP_LAYOUTS[node.op.type]
      transpose_order = OptimizeCommander._compute_transpose_order(src_layout, layout)
      return node.op.params[node.op.ParamName.WEIGHTS].data.copy().transpose(transpose_order)
    elif node.op.type in [NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D]:
      # [channel_multipier, H, W, I]
      weight = node.op.params[node.op.ParamName.WEIGHTS].data.copy()
      # [I, H, W, channel_multiplier]
      weight = weight.transpose(3, 1, 2, 0)
      # depthwise always put channel num as first dimension
      return weight
    elif node.op.type in [NNDCT_OP.DEPTHWISE_CONV3D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D]:
      weight = node.op.params[node.op.ParamName.WEIGHTS].data.copy()
      # [I, D, H, W, channel_multiplier]
      weight = weight.transpose(4, 1, 2, 3, 0)
      # depthwise always put channel num as first dimension
      return weight
  
  '''
  @staticmethod
  def _get_weight_data(node, layout):
    if node.op.type in [NNDCT_OP.CONV2D, NNDCT_OP.CONVTRANSPOSE2D]:
      if layout == "OHWI":
        return node.op.params[node.op.ParamName.WEIGHTS].data.copy()
      elif layout == "IHWO":
        return node.op.params[node.op.ParamName.WEIGHTS].data.copy().transpose(3, 1, 2, 0)
      else:
        raise ValueError("only support OHWI/IHWO layout for conv2d weight")
    elif node.op.type == NNDCT_OP.DEPTHWISE_CONV2D:
      # [channel_multipier, H, W, I]
      weight = node.op.params[node.op.ParamName.WEIGHTS].data.copy()
      # [I, H, W, channel_multiplier]
      weight = weight.transpose(3, 1, 2, 0)
      # depthwise always put channel num as first dimension
      return weight
  '''
  
  @staticmethod
  def _set_weight_data(node, data, layout):
    if node.op.type in [NNDCT_OP.CONV2D, NNDCT_OP.CONVTRANSPOSE2D, NNDCT_OP.CONV3D, NNDCT_OP.CONVTRANSPOSE3D]:
      dest_layout = _OP_LAYOUTS[node.op.type]
      transpose_order = OptimizeCommander._compute_transpose_order(layout, dest_layout)
      data = data.transpose(transpose_order)
      node.op.params[node.op.ParamName.WEIGHTS].from_ndarray(data)
    elif node.op.type in [NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D]:
      data = data.transpose(3, 1, 2, 0)
      node.op.params[node.op.ParamName.WEIGHTS].from_ndarray(data)
    elif node.op.type in [NNDCT_OP.DEPTHWISE_CONV3D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D]:
      data = data.transpose(4, 1, 2, 3, 0)
      node.op.params[node.op.ParamName.WEIGHTS].from_ndarray(data)
  
  '''
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
  '''
  
  @staticmethod
  def _get_bias_data(node):
    return node.op.params[node.op.ParamName.BIAS].data.copy()
  
  @staticmethod
  def _set_bias_data(node, data):
    node.op.params[node.op.ParamName.BIAS].from_ndarray(data)

  @staticmethod
  def _combine_weight_and_bias(weights_ihw, bias):
    if bias is not None: 
      bias_clamp = bias.copy().reshape(-1, 1)
      if np.count_nonzero(weights_ihw) != weights_ihw.size:
        #print(f"{group[0].name}' weight has zero element")
        weight_ihw_clamp = weights_ihw.copy()
        for channel in range(weight_ihw_clamp.shape[0]):
          if np.count_nonzero(weight_ihw_clamp[channel]) == 0:
            bias_clamp[channel] = 0.0
            weight_ihw_clamp[channel] = 1e-7
          elif np.count_nonzero(weight_ihw_clamp[channel]) != weight_ihw_clamp[channel].size:
            minval = np.min(np.fabs(np.ma.masked_where(weight_ihw_clamp[channel] == 0.0,
                                    weight_ihw_clamp[channel])))
            weight_ihw_clamp[channel] = np.where(weight_ihw_clamp[channel] == 0.0, 
                                                        minval, weight_ihw_clamp[channel])
          
        weight_ihw_clamp = np.where(np.fabs(weight_ihw_clamp) < 1e-7, 1e-7, weight_ihw_clamp)
        factor = np.fabs(bias_clamp) / np.fabs(weight_ihw_clamp) 
      else:
        weight_ihw_clamp = weights_ihw.copy()
        weight_ihw_clamp = np.where(np.fabs(weight_ihw_clamp) < 1e-7, 1e-7, weight_ihw_clamp)
        factor = np.fabs(bias_clamp) / np.fabs(weight_ihw_clamp)
      
      if (np.fabs(bias).max() < 10) and (np.fabs(bias).max()/np.fabs(weight_ihw_clamp).max() < 20):
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
      weight_bias = np.concatenate((weights_ihw, bias_clamp / shrink_factor), axis=1)
    else:
      weight_bias = weights_ihw
    
    return weight_bias

  @classmethod
  def _scale_weights_and_bias_with_conv_reluk(cls, group):
    ops_display_layout = {NNDCT_OP.CONV2D: "OHWI", NNDCT_OP.CONVTRANSPOSE2D: "OHWI", \
                          NNDCT_OP.DEPTHWISE_CONV2D: "OHWI", NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D: "OHWI", \
                          NNDCT_OP.CONV3D: "ODHWI", NNDCT_OP.CONVTRANSPOSE3D: "ODHWI",\
                          NNDCT_OP.DEPTHWISE_CONV3D: "ODHWI", NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D: "ODHWI"}
    ops_scale_layout = {NNDCT_OP.CONV2D: "IHWO", NNDCT_OP.CONVTRANSPOSE2D: "IHWO", \
                          NNDCT_OP.DEPTHWISE_CONV2D: "IHWO", NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D: "IHWO", \
                          NNDCT_OP.CONV3D: "IDHWO", NNDCT_OP.CONVTRANSPOSE3D: "IDHWO",\
                          NNDCT_OP.DEPTHWISE_CONV3D: "IDHWO", NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D: "IDHWO"}
    # conv: [O, H, W, I] / depthwise_conv: [I, H, W, channel_multiplier]
    conv_0_weight = cls._get_weight_data(group[0], layout=ops_display_layout[group[0].op.type])
    conv_0_bias = None
    if group[0].node_attr(group[0].op.AttrName.BIAS_TERM):
      conv_0_bias = cls._get_bias_data(group[0])
      
    if group[0].op.type in [NNDCT_OP.CONVTRANSPOSE2D] and group[0].op.attr['group'] > 1:
      group_num = group[0].op.attr['group']
      conv_0_weightt_list = np.split(conv_0_weight, group_num, axis=3)
      conv_0_weight = np.concatenate(conv_0_weightt_list, axis=0)
    elif group[0].op.type in [NNDCT_OP.CONVTRANSPOSE3D] and group[0].op.attr['group'] > 1:
      group_num = group[0].op.attr['group']
      conv_0_weightt_list = np.split(conv_0_weight, group_num, axis=4)
      conv_0_weight = np.concatenate(conv_0_weightt_list, axis=0)
    
    # conv2d: [O, H, W, I] / depthwise_conv2d: [I, H, W, channel_multiplier]
    # conv3d: [O, D, H, W, I] / depthwise_conv3d: [I, D, H, W, channel_multiplier]
    conv_1_weight = cls._get_weight_data(group[-1], layout=ops_display_layout[group[-1].op.type])
    if group[-1].op.type in [NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D]:
      # depthwise_conv:  [I, H, W, channel_multiplier] -> [channel_muliplier, H, W, I]
      conv_1_weight = conv_1_weight.transpose(3, 1, 2, 0)
    elif group[-1].op.type in [NNDCT_OP.DEPTHWISE_CONV3D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D]:
      # depthwise_conv3d:  [I, D, H, W, channel_multiplier] -> [channel_muliplier, D, H, W, I]
      conv_1_weight = conv_1_weight.transpose(4, 1, 2, 3, 0)
    elif group[-1].op.type in [NNDCT_OP.CONV2D] and group[-1].op.attr['group'] > 1:
      group_num = group[-1].op.attr['group']
      conv_1_weight_list = np.split(conv_1_weight, group_num, axis=0)
      conv_1_weight = np.concatenate(conv_1_weight_list, axis=3)
    elif group[-1].op.type in [NNDCT_OP.CONV3D] and group[-1].op.attr['group'] > 1:
      group_num = group[-1].op.attr['group']
      conv_1_weight_list = np.split(conv_1_weight, group_num, axis=0)
      conv_1_weight = np.concatenate(conv_1_weight_list, axis=4)
      
    # compute scale:
    conv_0_weight_ihw = conv_0_weight.reshape(conv_0_weight.shape[0], -1)
    conv_0_weight_bias = cls._combine_weight_and_bias(conv_0_weight_ihw, conv_0_bias)

    merge_dim = tuple(np.arange(conv_1_weight.ndim-1))
    range_0 = np.max(np.fabs(conv_0_weight_bias), axis=1)
    range_1 = np.max(np.fabs(conv_1_weight), axis=merge_dim)
    sqrt_of_ranges = np.sqrt(range_0 * range_1)
    scale = np.ones_like(range_0)
    
    scale = np.where(sqrt_of_ranges != 0, range_0 / sqrt_of_ranges, scale)
    
    i_max = np.max(np.fabs(conv_0_weight_bias), axis=1)
    o_max = np.max(np.fabs(conv_1_weight), axis=merge_dim)
    scale = np.where((i_max + o_max) < 0.5, 1, scale)
    
    scale_transpose = cls._compute_transpose_order(ops_display_layout[group[0].op.type], 
                                               ops_scale_layout[group[0].op.type])
    conv_0_weight = conv_0_weight.transpose(scale_transpose) / scale
    conv_0_weight = conv_0_weight.transpose(scale_transpose)
    if conv_0_bias is not None:
      conv_0_bias = conv_0_bias / scale
    
    if group[0].op.type in [NNDCT_OP.CONVTRANSPOSE2D] and group[0].op.attr['group'] > 1:
      group_num = group[0].op.attr['group']
      conv_0_weightt_list = np.split(conv_0_weight, group_num, axis=0)
      conv_0_weight = np.concatenate(conv_0_weightt_list, axis=3)
    elif group[0].op.type in [NNDCT_OP.CONVTRANSPOSE3D] and group[0].op.attr['group'] > 1:
      group_num = group[0].op.attr['group']
      conv_0_weightt_list = np.split(conv_0_weight, group_num, axis=0)
      conv_0_weight = np.concatenate(conv_0_weightt_list, axis=4)
      
    conv_1_weight = conv_1_weight * scale
    if group[-1].op.type in [NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D]:
      # depthwise_conv: [channel_muliplier, H, W, I] -> [I, H, W, channel_multiplier]
      conv_1_weight = conv_1_weight.transpose(3, 1, 2, 0) 
    elif group[-1].op.type in [NNDCT_OP.DEPTHWISE_CONV3D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D]:
      # depthwise_conv: [channel_muliplier, D, H, W, I] -> [I, D, H, W, channel_multiplier]
      conv_1_weight = conv_1_weight.transpose(4, 1, 2, 3, 0) 
    elif group[-1].op.type in [NNDCT_OP.CONV2D] and group[-1].op.attr['group'] > 1:
      group_num = group[-1].op.attr['group']
      conv_1_weight_list = np.split(conv_1_weight, group_num, axis=3)
      conv_1_weight = np.concatenate(conv_1_weight_list, axis=0)
    elif group[-1].op.type in [NNDCT_OP.CONV3D] and group[-1].op.attr['group'] > 1:
      group_num = group[-1].op.attr['group']
      conv_1_weight_list = np.split(conv_1_weight, group_num, axis=4)
      conv_1_weight = np.concatenate(conv_1_weight_list, axis=0)
    
    cls._set_weight_data(group[0], conv_0_weight, layout=ops_display_layout[group[0].op.type])
    if conv_0_bias is not None:
      cls._set_bias_data(group[0], conv_0_bias)
    
    cls._set_weight_data(group[-1], conv_1_weight, layout=ops_display_layout[group[-1].op.type])
    
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
