

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
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List

from nndct_shared.base import NNDCT_OP


def create_quant_strategy(bits_weight: int, bits_act: int, is_lstm: bool, mix_bit: bool):
  if is_lstm:
    quant_strategy = LstmQstrategy(bits_weight, bits_act, bits_act)
  else:
    if mix_bit:
      quant_strategy = TQTStrategy(bits_weight, 8, bits_act)
    else:
      quant_strategy = DefaultQstrategy(bits_weight, bits_act, bits_act)

  return quant_strategy


class QuantStrategyBase(ABC):

  def __init__(self, bits_weight, bits_bias, bits_activation, mix_bit=False):
    self._bits_weight = bits_weight
    self._bits_bias = bits_bias
    self._bits_act = bits_activation
    self._mix_bit = mix_bit

  @abstractmethod
  def create_quant_config(self, *args,
                          **kwargs) -> Dict[str, Dict[str, List[int]]]:
    """create input/output/param quantization configuration
    Returns
    dict: quant config
    """
    pass

  def _get_default_quant_config(self,
                                quant_info_mgr,
                                lstm=False):
    """
    1. unified activation bits
    2 .mixed bits for lstm 
    
    """
    # import ipdb
    # ipdb.set_trace()
    config = {'param': {}, 'output': {}, 'input': {}}
    for node in quant_info_mgr.Nndctgraph.nodes:
      # print('---- Handling node %s type: %s' % (node.name, node.op.type))
      if quant_info_mgr.is_node_quantizable(node, lstm):
        # parameters
        for k in quant_info_mgr.quant_node_params(node).keys():
          p = quant_info_mgr.quant_node_params(node)[k]
          # for mix precision quantization
          bw = self._bits_act
          if (node.has_bound_params() and 
            (hasattr(node.op.ParamName, 'WEIGHTS') and k == node.op.ParamName.WEIGHTS or
             hasattr(node.op.ParamName, 'GAMMA') and k == node.op.ParamName.GAMMA)):
            bw = self._bits_weight
          config['param'][p.name] = [bw, None]
          # print('---- Add fix of param %s' % p.name)
        # output blobs
        end = quant_info_mgr.quant_output(node.name).name
        if end not in config['output']:
          config['output'][end] = [self._bits_act, None]
          # print('---- Add fix of output blob %s' % end)
        # input blobs (for mix precision quantization)
        if self._bits_weight != self._bits_act:
          if node.op.type in [NNDCT_OP.DENSE, NNDCT_OP.CONV2D]:
            config['input'][node.name] = [self._bits_weight, None]
            # print('---- Add fix of input blob %s' % end)
      elif (lstm and (node in quant_info_mgr.Nndctgraph.inputs)):
        # print('---- Handling input node %s' % (node.name))
        # this path is only for quantizing a whole graph without quant stub OP
        # for lstm, check the following node type
        if (node.in_quant_part or (any(
            (quant_info_mgr.is_node_quantizable(c, lstm) and
             c.op.type is not NNDCT_OP.QUANT_STUB)
            for c in quant_info_mgr.Nndctgraph.children(node.name)))):
          end = quant_info_mgr.quant_output(node.name).name
          if end not in config['output']:
            config['output'][end] = [self._bits_act, None]
            # print('---- Add fix of quant net input blob %s' % end)
    
    # check the input fix of all quantized ops 
    # import ipdb
    # ipdb.set_trace()
    if not lstm:
      for node in quant_info_mgr.Nndctgraph.nodes:
        if quant_info_mgr.is_node_quantizable(node, lstm):
          #print('---- Check input of node %s type: %s' % (node.name, node.op.type))
          if node.op.type not in [NNDCT_OP.INPUT, NNDCT_OP.QUANT_STUB, NNDCT_OP.CONCAT]:
            for p_n in quant_info_mgr.Nndctgraph.parents(node):
              # if not quant_info_mgr.op_unquantizable(p_n.op.type):
                end = quant_info_mgr.quant_output(p_n.name).name
                end_node = quant_info_mgr.Nndctgraph.node(end)
                out_is_tensor = True
                for tensor in end_node.out_tensors:
                  if tensor.shape == None:
                    out_is_tensor = False
                if end not in config['output'] and out_is_tensor:
                  config['output'][end] = [self._bits_act, None]
                  #print('---- Add fix of output blob %s type: %s' % (end, end_node.op.type))
                  
          elif node.op.type in [NNDCT_OP.INPUT]:
            cn_nodes = quant_info_mgr.Nndctgraph.children(node)
            if len(cn_nodes) == 1 and cn_nodes[0].op.is_custom_op:
              end = quant_info_mgr.quant_output(node.name).name
              if end in config['output']:
                del config['output'][end]
                node.in_quant_part = False
              
    return config

  @property
  def num_bits_w(self):
    return self._bits_weight

  @property
  def num_bits_a(self):
    return self._bits_act
  
  @property
  def mix_bit(self):
    return self._mix_bit
  
  
class DefaultQstrategy(QuantStrategyBase):

  def __init__(self, bits_weight, bits_bias, bits_activation, mix_bit=False):
    super().__init__(bits_weight, bits_bias, bits_activation, mix_bit)

  def create_quant_config(self, quant_info_mgr):
    return self._get_default_quant_config(quant_info_mgr)


class LstmQstrategy(QuantStrategyBase):

  def __init__(self, bits_weight, bits_bias, bits_activation):
    super().__init__(bits_weight, bits_bias, bits_activation, False)

  def create_quant_config(self, quant_info_mgr):
    return self._get_default_quant_config(quant_info_mgr, lstm=True)


class TQTStrategy(QuantStrategyBase):
  
  _max_bit = 8
  _min_bit = 4
  
  def __init__(self, bits_weight, bits_bias, bits_activation):
    super().__init__(bits_weight, bits_bias, bits_activation, True)

    # [input_bits, output_bits]
    self._init_bit_config = {
        NNDCT_OP.CONV2D: [self._bits_act, self._bits_act],
        NNDCT_OP.ADD: [self._max_bit, self._max_bit],
        NNDCT_OP.MAX_POOL: [self._max_bit, self._max_bit],
        NNDCT_OP.AVG_POOL: [self._max_bit, self._max_bit],
        NNDCT_OP.ADAPTIVEAVGPOOL2D: [self._max_bit, self._max_bit],
        NNDCT_OP.DENSE: [self._bits_act, self._bits_act],
        NNDCT_OP.BATCH_NORM: [self._max_bit, self._min_bit],
        NNDCT_OP.QUANT_STUB: [None, self._max_bit]
    }

    self._input_fix_op_types = [
        NNDCT_OP.CONV2D, NNDCT_OP.CONVTRANSPOSE2D, NNDCT_OP.DENSE,
    ]
    
    self._activation_op_types = [NNDCT_OP.RELU, NNDCT_OP.RELU6, NNDCT_OP.TANH, NNDCT_OP.LEAKY_RELU]
    # self._passive_quant_ops = [NNDCT_OP.CONCAT]

  def _get_init_config_from_type(self, op_type):
    default = [self._max_bit, self._max_bit]
    return copy.copy(self._init_bit_config.get(op_type, default))

  def create_quant_config(self, quant_info_mgr):
    # [input_bw, output_bw]
    config = {
        "param": defaultdict(list),
        "output": defaultdict(list),
        "input": defaultdict(list)
    }
    # handle params bits
    for node in quant_info_mgr.Nndctgraph.nodes:
      # print('---- Handling node %s type: %s' % (node.name, node.op.type))
      if quant_info_mgr.is_node_quantizable(node, False):
        # parameters
        for k in quant_info_mgr.quant_node_params(node).keys():
          p = quant_info_mgr.quant_node_params(node)[k]
          # for mix precision quantization
          if k == node.op.ParamName.WEIGHTS:
            config['param'][p.name] = [self._bits_weight, None]
          else:
            config['param'][p.name] = [self._bits_bias, None]
          # print('---- Add fix of param %s' % p.name)

    # handle output bits
    node_bits_map = {}
    for node in quant_info_mgr.Nndctgraph.nodes:
      if quant_info_mgr.is_node_quantizable(node, False):
        # *_, end = quant_info_mgr.quant_groups[node.name]
        node_bits_map[node.name] = self._get_init_config_from_type(node.op.type)
        if node in (tensor.node for tensor in quant_info_mgr.Nndctgraph.end_tensors):
          node_bits_map[node.name][1] = self._max_bit
        elif node.op.type in self._input_fix_op_types:
          output_bit_list = []
          for c_node in quant_info_mgr.Nndctgraph.children(node):
            self._find_next_quant_nodes_bits(quant_info_mgr, c_node,
                                             output_bit_list)
          node_bits_map[node.name][1] = max(output_bit_list) if output_bit_list else self._max_bit
          # if node.op.type in self._passive_quant_ops:
          #   node_bits_map[node.name][0] = node_bits_map[node.name][1]

        for pn in quant_info_mgr.Nndctgraph.parents(node):
          if pn.name in node_bits_map:
            p_out_bits = node_bits_map[pn.name][1]
            if p_out_bits == node_bits_map[node.name][0]:
              node_bits_map[node.name][0] = None

      else:
        for pn in quant_info_mgr.Nndctgraph.parents(node):
          if pn.name in node_bits_map:
            node_bits_map[node.name] = node_bits_map[pn.name]
            break
          
        if node.name not in node_bits_map:
          node_bits_map[node.name] = self._get_init_config_from_type(node.op.type)
       

    # handle input bits
    for node in quant_info_mgr.Nndctgraph.nodes:
      if quant_info_mgr.is_node_quantizable(node, False):
        *_, end = quant_info_mgr.quant_groups[node.name]
        if node.op.type in self._input_fix_op_types and node_bits_map[node.name][0] is not None:
          config["input"][node.name] = [node_bits_map[node.name][0], None]
          
        if end not in config["output"] and node_bits_map[node.name][1] is not None:
          quant_output = None
          for out_node in quant_info_mgr.quant_groups[node.name]:
            if quant_info_mgr.Nndctgraph.node(out_node).op_type in self._activation_op_types:
              quant_output = out_node
              break
          if quant_output is not None:
            config["output"][quant_output] = [node_bits_map[node.name][1], None]
          else:
            config["output"][node.name] = [node_bits_map[node.name][1], None]
          
    # import json
    # string = json.dumps(config, indent=4, separators=(',', ': '))
    # print(string)
    return config

  def _find_next_quant_nodes_bits(self,
                                  quant_info_mgr,
                                  node,
                                  output_bits_candidates=None):
    if quant_info_mgr.is_node_quantizable(node, False):
      output_bits = self._get_init_config_from_type(node.op.type)[0]
      output_bits_candidates.append(output_bits)
      return

    for c_node in quant_info_mgr.Nndctgraph.children(node):
      self._find_next_quant_nodes_bits(quant_info_mgr, c_node,
                                       output_bits_candidates)
