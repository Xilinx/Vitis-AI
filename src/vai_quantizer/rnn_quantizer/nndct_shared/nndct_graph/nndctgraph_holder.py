
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
        NNDCT_OP.BATCH_NORM, NNDCT_OP.BATCH_NORM1D, NNDCT_OP.BATCH_NORM3D,
        NNDCT_OP.BIAS_ADD, NNDCT_OP.BASIC_LSTM, NNDCT_OP.BASIC_GRU,
        NNDCT_OP.CONV2D, NNDCT_OP.CONCAT, NNDCT_OP.DEPTHWISE_CONV2D,
        NNDCT_OP.CONV1D,
        NNDCT_OP.DENSE, NNDCT_OP.ADD, NNDCT_OP.MULTIPLY, NNDCT_OP.DIV,
        NNDCT_OP.MAX_POOL, NNDCT_OP.MAX, NNDCT_OP.MEAN,
        NNDCT_OP.MAX_POOL1D,
        NNDCT_OP.MIN, NNDCT_OP.RESIZE, NNDCT_OP.TANH,
        NNDCT_OP.SUB, NNDCT_OP.RSUB, NNDCT_OP.PAD, NNDCT_OP.QUANT_STUB,
        NNDCT_OP.INPUT, NNDCT_OP.CONV3D, NNDCT_OP.DEPTHWISE_CONV3D, NNDCT_OP.RESIZE_3D,
        NNDCT_OP.CONVTRANSPOSE3D, NNDCT_OP.SUM, NNDCT_OP.HSWISH, NNDCT_OP.HSIGMOID,
        NNDCT_OP.MATMUL, #,NNDCT_OP.CHANNEL_SCALE
        NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D, 
        NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D
    ]
    self.LSTM_QUANTIZABLE_OPS = [
        #NNDCT_OP.PLACEHOLDER,
        NNDCT_OP.CONV2D, NNDCT_OP.DENSE,
        NNDCT_OP.ADD, NNDCT_OP.MULTIPLY,
        NNDCT_OP.SIGMOID, NNDCT_OP.TANH,
        NNDCT_OP.SUB, NNDCT_OP.RSUB,
        NNDCT_OP.CONCAT, 
        NNDCT_OP.INPUT,
        NNDCT_OP.MATMUL, 
        NNDCT_OP.ADDMM
    ]
    self.QUANTIZABLE_OPS_WITH_PARAMS = [
        NNDCT_OP.DENSE,
        NNDCT_OP.CONV1D,
        NNDCT_OP.CONV2D, NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.CONVTRANSPOSE2D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D,
        NNDCT_OP.CONV3D, NNDCT_OP.DEPTHWISE_CONV3D, NNDCT_OP.CONVTRANSPOSE3D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D,
        NNDCT_OP.BATCH_NORM, NNDCT_OP.BATCH_NORM1D, NNDCT_OP.BATCH_NORM3D, 
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
    self.QUANTIZABLE_DTYPES = ['float32', 'float64']
    self.DPU_APPROXIMATION_OPS = [
        NNDCT_OP.AVG_POOL, NNDCT_OP.ADAPTIVEAVGPOOL2D,
        NNDCT_OP.SIGMOID, NNDCT_OP.TANH,
        NNDCT_OP.HSWISH, NNDCT_OP.HSIGMOID
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
    
    self.SOFT_FUSED_OPS = [
      # NNDCT_OP.CLAMP,
      NNDCT_OP.HARDTANH,
      NNDCT_OP.RELU,
      NNDCT_OP.RELU6,
      NNDCT_OP.RELUK,
      NNDCT_OP.CHANNEL_SCALE,
      NNDCT_OP.FLATTEN,
      NNDCT_OP.SQUEEZE,
      NNDCT_OP.PIXEL_SHUFFLE,
      NNDCT_OP.RESHAPE,
      NNDCT_OP.SPLIT,
      NNDCT_OP.TRANSPOSE,
      NNDCT_OP.DROPOUT,
      NNDCT_OP.CONTIGUOUS,
      NNDCT_OP.PERMUTE,
      NNDCT_OP.EXPAND,
      NNDCT_OP.INPLACE_COPY,
      NNDCT_OP.REPEAT,
      # NNDCT_OP.SELECT,
      NNDCT_OP.UNSQUEEZE,
      
    ]
      
    

  def get_model_type(self):
    return self.model_type or 'Nndct'
 
  def get_Nndctnode(self, node_name=None, params=None, idx=None, inputs=None):
    #TODO: use parameters to find node, use normal parameters
    def __from_node_name():
      for node in self.__Nndctgraph.nodes:
        if node_name == node.name:
          return node
        elif node_name in node.alias:
          #TODO: is taht safe?
          return node
      raise KeyError(
          "{} do not exist in Graph, please check!".format(node_name))

    def __from_params():
      for node in self.__Nndctgraph.nodes:
        valid_params = node.op.params
        if all(p in valid_params for p in params):
          return node
      raise KeyError(
          "node with params {} do not exist in Graph, please check!".format(
              params))

    def __from_inputs():
      mapped_inputs = [
          nndct_utils.node_from_output(i, self.get_model_type()) for i in inputs
      ]
      for node in self.__Nndctgraph.nodes:
        if all(i in node.inputs for i in mapped_inputs):
          return node
      raise KeyError(
          "node with inputs {}(map to {}) do not exist in Graph, please check!"
          .format(inputs, mapped_inputs))

    def _from_idx():
      for node in self.__Nndctgraph.nodes:
        if node.idx == idx:
          return node
      raise KeyError(
          "node with idx {} do not exist in Graph, please check!".format(idx))

    if node_name is not None:
      return __from_node_name()
    elif idx is not None:
      return _from_idx()
    elif params and len(params) > 0:
      return __from_params()
    elif inputs and len(inputs) > 0:
      return __from_inputs()
    else:
      raise Exception(
          "One of node name,params and inputs should be given to locate Xnode in the graph"
      )

  def is_node_quantizable(self, node, lstm):
    if not lstm:
      # check the node type if it needs to be quantized
      ret = node.op.type in self.QUANTIZABLE_OPS
      # check the node is in quant_stub or not:
      ret = ret and node.in_quant_part
      # if CONST input is used to caluation
      if node.in_quant_part and node.op.type == NNDCT_OP.CONST:
        for c in self.Nndctgraph.children(node):
          if c.op.type in [NNDCT_OP.ADD, NNDCT_OP.SUB, NNDCT_OP.MULTIPLY, NNDCT_OP.DIV]:
            ret = True
      return ret
    else:
      return node.op.type in self.LSTM_QUANTIZABLE_OPS

  def node_output_quantizable(self, node):
    if node.op.type in self.MULTIPLE_OUTPUTS_OPS:
      return False
    else:
      return True
    
  def op_unquantizable(self, op_type):
    # The output of op is scalar
    return False if op_type in self.SOFT_FUSED_OPS else True
      
    
  def node_quantizable_with_params(self, node):
    if node.op.type in self.QUANTIZABLE_OPS_WITH_PARAMS:
      return True
    else:
      return False

  def quant_node_params(self, node_or_name):
    node = self._find_node(node_or_name)
    if node.op.type in [NNDCT_OP.BATCH_NORM, NNDCT_OP.BATCH_NORM1D, NNDCT_OP.BATCH_NORM3D]:
      return {
          k: v
          for k, v in node.op.params.items()
          if k in [node.op.ParamName.GAMMA, node.op.ParamName.BETA]
      }
    else:
      return node.op.params

  def quant_start_node(self, node_or_name):
    node = self._find_node(node_or_name)
    return self.__Nndctgraph.node(self._QuantGroups[node.name][0])

  def quant_is_start_node(self, node_or_name):
    node = self._find_node(node_or_name)
    return node == self.__Nndctgraph.node(self._QuantGroups[node.name][0])

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
    return end_node

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

  def quant_params(self, node_or_name):
    node = self._find_node(node_or_name)
    return list(
        chain.from_iterable(
            [self.node(n).op.params for n in self._QuantGroups[node.name]]))


  def _find_node(self, node_or_name):
    if isinstance(node_or_name, str):
      return self.get_Nndctnode(node_name=node_or_name)
    else:
      return node_or_name

  def is_conv_like(self, node_or_name):
    node = self._find_node(node_or_name)
    return node.op.type in self.CONV_LIKE_OPS

  @property
  def quant_groups(self):
    return self._QuantGroups

  @property
  def Nndctgraph(self):
    return self.__Nndctgraph

  @property
  def dpu_approximation_ops(self):
    return self.DPU_APPROXIMATION_OPS

