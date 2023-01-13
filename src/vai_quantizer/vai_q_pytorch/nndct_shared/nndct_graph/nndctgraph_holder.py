
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
from itertools import chain
from functools import reduce

from nndct_shared.base import NNDCT_KEYS, NNDCT_OP, NNDCT_PARAM
from nndct_shared.utils.log import NndctDebugger
from nndct_shared import utils as nndct_utils

from .base_graph import Graph
from .base_node import Node
from .base_operator import Operation

from .utils import *

class NndctGraphHolder(NndctDebugger):
  
  

  def __init__(self, graph=None, model_type=None): 
    self.__Nndctgraph = graph
    self.model_type = model_type
    self.scan_commander = {}
    self._QuantGroups = None

    #for quantization
    self.QUANTIZABLE_OPS = [
        NNDCT_OP.AVG_POOL, NNDCT_OP.ADAPTIVEAVGPOOL2D, NNDCT_OP.CONVTRANSPOSE2D,
        NNDCT_OP.BATCH_NORM,NNDCT_OP.LAYER_NORM,NNDCT_OP.INSTANCE_NORM,NNDCT_OP.GROUP_NORM,
        NNDCT_OP.BIAS_ADD, NNDCT_OP.BASIC_LSTM, NNDCT_OP.BASIC_GRU,
        NNDCT_OP.CONV2D, NNDCT_OP.CONCAT, NNDCT_OP.DEPTHWISE_CONV2D,
        NNDCT_OP.CONV1D,
        NNDCT_OP.DENSE, NNDCT_OP.ADD, NNDCT_OP.MULTIPLY, NNDCT_OP.DIV,
        NNDCT_OP.MAX_POOL, NNDCT_OP.MAX, NNDCT_OP.MEAN,
        NNDCT_OP.MAX_POOL1D,
        NNDCT_OP.MIN, NNDCT_OP.RESIZE, 
        NNDCT_OP.SUB, NNDCT_OP.RSUB, NNDCT_OP.PAD, NNDCT_OP.QUANT_STUB,
        NNDCT_OP.INPUT, NNDCT_OP.CONV3D, NNDCT_OP.DEPTHWISE_CONV3D, NNDCT_OP.RESIZE_3D,
        NNDCT_OP.CONVTRANSPOSE3D, NNDCT_OP.SUM, NNDCT_OP.HSWISH, NNDCT_OP.HSIGMOID,
        NNDCT_OP.MATMUL, 
        NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D, 
        NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D,
        NNDCT_OP.CORRELATION1D_ELEMWISE,
        NNDCT_OP.CORRELATION2D_ELEMWISE,
        NNDCT_OP.COST_VOLUME,
        NNDCT_OP.SOFTMAX,
        NNDCT_OP.PRELU,
        # NNDCT_OP.CLAMP,
        NNDCT_OP.GELU,
        NNDCT_OP.MISH,
    ]
    self.TENSORRT_QUANTIZABLE_OPS = [
        NNDCT_OP.AVG_POOL, NNDCT_OP.ADAPTIVEAVGPOOL2D, NNDCT_OP.CONV1D, 
        NNDCT_OP.CONV2D, NNDCT_OP.DEPTHWISE_CONV2D, 
        NNDCT_OP.CONV3D, NNDCT_OP.DEPTHWISE_CONV3D,
        NNDCT_OP.CONVTRANSPOSE2D, NNDCT_OP.CONVTRANSPOSE3D, 
        NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D, 
        NNDCT_OP.DENSE
    ]
   
    self.LSTM_QUANTIZABLE_OPS = [
        NNDCT_OP.CONV2D, NNDCT_OP.DENSE,
        NNDCT_OP.ADD, NNDCT_OP.MULTIPLY,
        NNDCT_OP.SIGMOID, NNDCT_OP.TANH,
        NNDCT_OP.SUB, NNDCT_OP.RSUB,
        NNDCT_OP.CONCAT, 
        NNDCT_OP.STACK,
        NNDCT_OP.INPUT,
        NNDCT_OP.MATMUL, 
        NNDCT_OP.ADDMM,
        NNDCT_OP.SOFTMAX,
        NNDCT_OP.LOG_SOFTMAX,
        NNDCT_OP.LAYER_NORM,
        NNDCT_OP.EMBEDDING
    ]
    self.QUANTIZABLE_OPS_WITH_PARAMS = [
        NNDCT_OP.DENSE,
        NNDCT_OP.CONV1D,
        NNDCT_OP.CONV2D, NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.CONVTRANSPOSE2D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D,
        NNDCT_OP.CONV3D, NNDCT_OP.DEPTHWISE_CONV3D, NNDCT_OP.CONVTRANSPOSE3D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D,
        NNDCT_OP.BATCH_NORM, 
        # NNDCT_OP.INSTANCE_NORM, 
    ]
    self.MULTIPLE_OUTPUTS_OPS = [ # OP types where cannot do quantization 
        NNDCT_OP.CHUNK, # has multiple outputs and cannot be deployed
        NNDCT_OP.STRIDED_SLICE, # copy part of memory from a tensor and create a new tensor
        NNDCT_OP.PERMUTE, # can not be deployed
        NNDCT_OP.FLATTEN, # no calculation and only tensor in-place operation
        NNDCT_OP.RESHAPE, # no calculation and only tensor in-place operation
        NNDCT_OP.PIXEL_SHUFFLE, # no calculation and only tensor in-place operation
        NNDCT_OP.CONTIGUOUS, # no calculation and only tensor in-place operation
        NNDCT_OP.SQUEEZE, # no calculation and only tensor in-place operation
        NNDCT_OP.UNSQUEEZE
    ]

    self.CONV_LIKE_OPS = [
        NNDCT_OP.CONV2D,
        NNDCT_OP.CONV3D,
        NNDCT_OP.CONV1D,
        NNDCT_OP.CONVTRANSPOSE2D,
        NNDCT_OP.CONVTRANSPOSE3D,
        NNDCT_OP.DEPTHWISE_CONV2D,
        NNDCT_OP.DEPTHWISE_CONV3D,
        NNDCT_OP.DENSE,
        NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D
    ]
    self.QUANTIZABLE_DTYPES = ['float16', 'float32', 'float64']
 
    
    
  def get_model_type(self):
    return self.model_type or 'Nndct'
 
  def get_Nndctnode(self, node_name):
    return self.__Nndctgraph.node(node_name)

  def is_node_quantizable(self, node, lstm):
    if not lstm:
      # check the node type if it needs to be quantized
      ret = node.op.type in self.QUANTIZABLE_OPS
      # check the node is in quant_stub or not:
      ret = ret and node.in_quant_part
      return ret
    else:
      ret= node.op.type in self.LSTM_QUANTIZABLE_OPS
      ret = ret and node.in_quant_part
      ret = ret and (not self.will_merge_with_table(node, lstm))
      return ret
    
  def is_node_tensorrt_quantizable(self, node):
    # check the node type if it needs to be quantized
    ret = node.op.type in self.TENSORRT_QUANTIZABLE_OPS
     # check the node is in quant_stub or not:
    ret = ret and node.in_quant_part
    return ret

  def node_output_quantizable(self, node):
    if node.op.type in self.MULTIPLE_OUTPUTS_OPS:
      return False
    else:
      return True
      
    
  def node_quantizable_with_params(self, node):
    if node.op.type in self.QUANTIZABLE_OPS_WITH_PARAMS:
      return True
    else:
      return False

  def quant_node_params(self, node_or_name):
    node = self._find_node(node_or_name)
    if node.op.type == NNDCT_OP.BATCH_NORM:
      return {
          k: v
          for k, v in node.op.params.items()
          if k in [node.op.ParamName.GAMMA, node.op.ParamName.BETA]
      }
    else:
      return node.op.params



  def is_concat_input(self, node_or_name):
    node = self._find_node(node_or_name)
    isConcatInput = False # only ouput tensor to concat
    children = self.Nndctgraph.children(node)
    if len(children) == 1 and children[0].op.type == NNDCT_OP.CONCAT:
      isConcatInput = True
    return isConcatInput

  def quant_output(self, node_or_name):
    node = self._find_node(node_or_name)
    if not node.in_quant_part:
      return node
    idx = -1
    end_node = self.__Nndctgraph.node(self._QuantGroups[node.name][idx])
    if self.is_concat_input(end_node):
      for c in self.Nndctgraph.children(end_node):
        if c.op.type == NNDCT_OP.CONCAT:
          return c
   
    while end_node.op.type in self.MULTIPLE_OUTPUTS_OPS:
      idx = idx - 1
      if -idx > len(self._QuantGroups[node.name]):
        break
      end_node = self.__Nndctgraph.node(self._QuantGroups[node.name][idx])
    while end_node.op.type in self.MULTIPLE_OUTPUTS_OPS:
      if not end_node.in_nodes:
          break
      up_node = end_node.in_nodes[0]
      end_node = self.__Nndctgraph.node(up_node)
    
    return end_node
  
  def quant_group(self, node_or_name):
    node = self._find_node(node_or_name)
    if not node.in_quant_part:
      return None
    QuantGroupTypes = []
    for node_name in self._QuantGroups[node.name]:
      QuantGroupTypes.append(self.__Nndctgraph.node(node_name).op.type)
    return self._QuantGroups[node.name], QuantGroupTypes

  # only used in TF RNN case
  def quant_input_names(self, node_or_name, inputs, params=None, validate=True):
    node = self._find_node(node_or_name)
    valid_inputs = None
    for i in inputs:
      try:
        input_node = self.get_Nndctnode(
            node_name=nndct_utils.node_from_output(i, self.get_model_type()))
      except KeyError:
        continue
      if input_node.op.type == NNDCT_OP.INPUT:
        valid_input = input_node.name
        if valid_inputs is None:
          valid_inputs = []
        valid_inputs.append(valid_input)

    return valid_inputs



  def _find_node(self, node_or_name):
    if isinstance(node_or_name, str):
      return self.get_Nndctnode(node_name=node_or_name)
    else:
      return node_or_name

  def is_conv_like(self, node_or_name):
    node = self._find_node(node_or_name)
    return node.op.type in self.CONV_LIKE_OPS

  def will_merge_with_table(self, node_or_name, lstm):
    if not lstm:
      return False
    node = self._find_node(node_or_name)
    if node.op.type in [NNDCT_OP.LAYER_NORM]:
      children = self.Nndctgraph.children(node)
      if len(children) == 1:
        if children[0].op.type in [NNDCT_OP.SIGMOID, NNDCT_OP.TANH]:
          return True
    return False

  @property
  def quant_groups(self):
    return self._QuantGroups

  @property
  def Nndctgraph(self):
    return self.__Nndctgraph


