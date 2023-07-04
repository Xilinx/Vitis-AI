
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
from typing import Dict, List, Union

from nndct_shared.base import NNDCT_OP
from nndct_shared.quantization import QuantConfigImpBase

class QuantStrategyBase(ABC):

  def __init__(self, quant_strategy_info: Dict[str, Union[str, int, bool]], 
               quant_config_imp: QuantConfigImpBase, is_lstm: bool):
    self._lstm = is_lstm
    self._quant_strategy_info = quant_strategy_info
    self._quant_config_imp = quant_config_imp

  @abstractmethod
  def create_quant_config(self, *args,
                          **kwargs) -> Dict[str, Dict[str, List[int]]]:
    """create input/output/param quantization configuration
    Returns
    dict: quant config
    """
    pass
    
  @property
  def quant_strategy_info(self):
    return  self._quant_strategy_info

  @property
  def lstm(self):
    return self._lstm
  
  @property
  def num_bits_w(self):
    return self._quant_strategy_info['weights']['bit_width']

  @property
  def num_bits_b(self):
    return self._quant_strategy_info['bias']['bit_width']

  @property
  def num_bits_a(self):
    return self._quant_strategy_info['activation']['bit_width']
  
  @property
  def mix_bit(self):
    return self._quant_strategy_info['mix_bit']

class TQTStrategy(QuantStrategyBase):
  
  _max_bit = 8
  _min_bit = 4
  
  def __init__(self, quant_strategy_info: Dict[str, Union[str, int, bool]], 
               quant_config_imp: QuantConfigImpBase):
    super().__init__(quant_strategy_info, quant_config_imp, False)
    self._bits_act = quant_strategy_info['activation']['bit_width']
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
    # handle params bits
    self._quant_config_imp.clear_quant_config()
    for node in quant_info_mgr.Nndctgraph.all_nodes():
      # print('---- Handling node %s type: %s' % (node.name, node.op.type))
      if quant_info_mgr.is_node_quantizable(node, False):
        # parameters
        for k in quant_info_mgr.quant_node_params(node).keys():
          p = quant_info_mgr.quant_node_params(node)[k]
          if p.name not in self._quant_config_imp.quant_config['param'].keys():
            # for mix precision quantization
            if k == node.op.ParamName.WEIGHTS:
              self._quant_config_imp.add_quant_config(p.name, self.num_bits_w, 'param')
            else:
              self._quant_config_imp.add_quant_config(p.name, self.num_bits_b, 'param')
            # print('---- Add fix of param %s' % p.name)

    # handle output bits
    node_bits_map = {}
    for node in quant_info_mgr.Nndctgraph.all_nodes():
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
    for node in quant_info_mgr.Nndctgraph.all_nodes():
      if quant_info_mgr.is_node_quantizable(node, False):
        *_, end = quant_info_mgr.quant_groups[node.name]
        if node.op.type in self._input_fix_op_types and node_bits_map[node.name][0] is not None:
          for tensor in node.in_tensors:
            if tensor.name not in self._quant_config_imp.quant_config['param'].keys():
              self._quant_config_imp.add_quant_config(node.name, node_bits_map[node.name][0], 'input')
          
        if end not in self._quant_config_imp.quant_config["output"] and node_bits_map[node.name][1] is not None:
          quant_output = None
          for out_node in quant_info_mgr.quant_groups[node.name]:
            if quant_info_mgr.Nndctgraph.node(out_node).op_type in self._activation_op_types:
              quant_output = out_node
              break
          if quant_output is not None:
            for tensor in quant_info_mgr.Nndctgraph.node(quant_output).out_tensors:
              if tensor.name not in self._quant_config_imp.quant_config['param'].keys():
                self._quant_config_imp.add_quant_config(quant_output, node_bits_map[node.name][1], 'output')
          else:
            for tensor in node.out_tensors:
              if tensor.name not in self._quant_config_imp.quant_config['param'].keys():
                self._quant_config_imp.add_quant_config(node.name, node_bits_map[node.name][1], 'output')
          
    # import json
    # string = json.dumps(config, indent=4, separators=(',', ': '))
    # print(string)
    return self._quant_config_imp.quant_config, None

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
