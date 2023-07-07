

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

from abc import abstractmethod
from typing import Dict, Union
from nndct_shared.base import NNDCT_OP

from nndct_shared.utils import NndctOption
from nndct_shared.quantization import QuantStrategyBase
from pytorch_nndct.quantization import create_quant_algo
from pytorch_nndct.utils.nndct2torch_op_map import get_nndct_op_type, get_torch_op_type

class CPUGPUQstrategy(QuantStrategyBase):
  
  def __init__(self, quant_strategy_info: Dict[str, Union[str, int, bool]], is_lstm=False):
    super().__init__(quant_strategy_info, is_lstm=is_lstm)
    self._layer_quant_types = []
    self._layer_quant_names = []
    # for layer_type, layer_config in self._quant_strategy_info["layer_type_config"].items():
    #   nndct_layer_type = get_nndct_op_type(layer_type)
    #   self._layer_quant_types.append(nndct_layer_type)
  
  @abstractmethod
  def create_quant_config(self, quant_info_mgr):
    pass
  
  def get_layer_weights_config(self, node):
    layer_weights_quant = None
    if node.op.type in self._layer_quant_types:
      torch_type = get_torch_op_type(node.op.type)
      layer_weights_quant = self._quant_strategy_info["layer_type_config"][torch_type].get("weights", None)
    if node.name in self._layer_quant_names:
      layer_weights_quant = self._quant_strategy_info["layer_name_config"][node.name].get("weights", None)
    return layer_weights_quant
  
  def get_layer_bias_config(self, node):
    layer_bias_quant = None
    if node.op.type in self._layer_quant_types:
      torch_type = get_torch_op_type(node.op.type)
      layer_bias_quant = self._quant_strategy_info["layer_type_config"][torch_type].get("bias", None)
    if node.name in self._layer_quant_names:
      layer_bias_quant = self._quant_strategy_info["layer_name_config"][node.name].get("bias", None)
    return layer_bias_quant

class NndctCGQstrategy(CPUGPUQstrategy):

  def create_quant_config(self, quant_info_mgr):
    config = {'param': {}, 'output': {}, 'input': {}}
    quant_algo = {'param': {}, 'output': {}, 'input': {}}
    
    for layer_type, layer_config in self._quant_strategy_info["layer_type_config"].items():
      nndct_layer_type = get_nndct_op_type(layer_type)
      self._layer_quant_types.append(nndct_layer_type)
      
    for layer_name, layer_config in self._quant_strategy_info["layer_name_config"].items():
      self._layer_quant_names.append(layer_name)
    
    for node in quant_info_mgr.Nndctgraph.all_nodes():
      if quant_info_mgr.is_node_quantizable(node, self.lstm):
        # parameters
        for k in quant_info_mgr.quant_node_params(node).keys():
          p = quant_info_mgr.quant_node_params(node)[k]
          bw = self.num_bits_a
          # for mix precision quantization
          if (node.has_bound_params()):
            if (hasattr(node.op.ParamName, 'WEIGHTS') and k == node.op.ParamName.WEIGHTS or
             hasattr(node.op.ParamName, 'GAMMA') and k == node.op.ParamName.GAMMA):
              layer_weight_quant = self.get_layer_weights_config(node)
              if layer_weight_quant:
                bw = layer_weight_quant['bit_width']
                quant_algo['param'][p.name] = [create_quant_algo("weights", layer_weight_quant, node)]
              else:
                bw = self.num_bits_w
                quant_algo['param'][p.name] = [create_quant_algo("weights", self._quant_strategy_info["weights"], node)]
            elif (hasattr(node.op.ParamName, 'BIAS') and k == node.op.ParamName.BIAS or
             hasattr(node.op.ParamName, 'BETA') and k == node.op.ParamName.BETA):
              layer_bias_quant = self.get_layer_bias_config(node)
              if layer_bias_quant:
                bw = layer_bias_quant['bit_width']
                quant_algo['param'][p.name] = [create_quant_algo("bias", layer_bias_quant, node)]
              else:
                bw = self.num_bits_b
                quant_algo['param'][p.name] = [create_quant_algo("bias", self._quant_strategy_info["bias"], node)]
          # TODO: use set function set config
          config['param'][p.name] = [[bw, None, None, None]] # bitwidth, scale, zero_point
         
        # output blobs
        end = quant_info_mgr.quant_output(node.name).name
        if end not in config['output']:
          layer_group_quant = self.get_quant_group_activation_config(quant_info_mgr, node.name)
          config['output'][end] = []
          quant_algo['output'][end] = []
          for tensor in quant_info_mgr.quant_output(node.name).out_tensors:
            if tensor.name not in config['param'].keys():
              if layer_group_quant:
                config['output'][end].append([layer_group_quant['bit_width'], None, None, None])
                quant_algo['output'][end].append(create_quant_algo("activation", layer_group_quant, node))
              else:
                config['output'][end].append([self.num_bits_a, None, None, None])
                quant_algo['output'][end].append(create_quant_algo("activation", self._quant_strategy_info["activation"], node))

        # input blobs (for mix precision quantization)
        if self.num_bits_w != self.num_bits_a:
          if node.op.type in [NNDCT_OP.DENSE, NNDCT_OP.CONV2D]:
            layer_weight_quant = self.get_layer_weights_config(node)
            config['input'][node.name] = []
            quant_algo['input'][node.name] = []
            for tensor in node.in_tensors:
              if tensor.name not in config['param'].keys():
                if layer_weight_quant:
                  config['input'][node.name].append([layer_weight_quant['bit_width'], None, None, None])
                  quant_algo['input'][node.name].append(create_quant_algo("weights", layer_weight_quant, node))
                else:
                  config['input'][node.name].append([self.num_bits_w, None, None, None])
                  quant_algo['input'][node.name].append(create_quant_algo("weights", self._quant_strategy_info["weights"], node))
      elif (self.lstm and (node in quant_info_mgr.Nndctgraph.inputs) and node.op.type not in [NNDCT_OP.BLOCK, NNDCT_OP.TUPLE_INPUT]):
        # this path is only for quantizing a whole graph without quant stub OP
        # for lstm, check the following node type
        if (node.in_quant_part or (any(
            (quant_info_mgr.is_node_quantizable(c, self.lstm) and
             c.op.type is not NNDCT_OP.QUANT_STUB)
            for c in quant_info_mgr.Nndctgraph.children(node.name)))):
          end = quant_info_mgr.quant_output(node.name).name
          if end not in config['output']:
            layer_group_quant = self.get_quant_group_activation_config(quant_info_mgr, node.name)
            config['output'][end] = []
            quant_algo['output'][end] = []
            config['output'][end] = []
            for tensor in quant_info_mgr.quant_output(node.name).out_tensors:
              if tensor.name not in config['param'].keys():
                #config['output'][end].append([self.num_bits_a, None])
                if layer_group_quant:
                  config['output'][end].append([layer_group_quant['bit_width'], None, None, None])
                  quant_algo['output'][end].append(create_quant_algo("activation", layer_group_quant, node))
                else:
                  config['output'][end].append([self.num_bits_a, None, None, None])
                  quant_algo['output'][end].append(create_quant_algo("activation", self._quant_strategy_info["activation"], node))
    
    # check the input fix of all quantized ops 
    if not self.lstm:
      for node in quant_info_mgr.Nndctgraph.all_nodes():
        if quant_info_mgr.is_node_quantizable(node, self.lstm):
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
                  layer_group_quant = self.get_quant_group_activation_config(quant_info_mgr, p_n.name)
                  config['output'][end] = []
                  quant_algo['output'][end] = []
                  for tensor in quant_info_mgr.quant_output(p_n.name).out_tensors:
                    if tensor.name not in config['param'].keys():
                      if layer_group_quant:
                        config['output'][end].append([layer_group_quant['bit_width'], None, None, None])
                        quant_algo['output'][end].append(create_quant_algo("activation", layer_group_quant, node))
                      else:
                        config['output'][end].append([self.num_bits_a, None, None, None])
                        quant_algo['output'][end].append(create_quant_algo("activation", self._quant_strategy_info["activation"], node))    
                  
          elif node.op.type in [NNDCT_OP.INPUT]:
            cn_nodes = quant_info_mgr.Nndctgraph.children(node)
            if len(cn_nodes) == 1 and cn_nodes[0].op.is_custom_op:
              end = quant_info_mgr.quant_output(node.name).name
              if end in config['output']:
                del config['output'][end]
                node.in_quant_part = False
              
    return config, quant_algo
  
  def get_quant_group_activation_config(self, quant_info_mgr, node_or_name):
    layer_activation_config = None
    layer_name_config = None
    quant_group_name, quant_group_types = quant_info_mgr.quant_group(node_or_name)
    if len(quant_group_types) == 0:
      return None
    for layer_type in self._layer_quant_types:
      if layer_type in quant_group_types:
        torch_type = get_torch_op_type(layer_type)
        if layer_activation_config == None:
          layer_activation_config = self._quant_strategy_info["layer_type_config"][torch_type].get("activation", None)
        else:
          config_temp = self._quant_strategy_info["layer_type_config"][torch_type].get("activation", None)
          if config_temp:
            if not CPUGPUQstrategy.config_equal(layer_activation_config, config_temp):
              raise ValueError("Can not set different activation quant configs in the group {}".format(quant_group_name))
    
    for layer_name in self._layer_quant_names:
      if layer_name in quant_group_name:
        if layer_name_config == None:
          layer_name_config = self._quant_strategy_info["layer_name_config"][layer_name].get("activation", None)
        else:
          config_temp = self._quant_strategy_info["layer_name_config"][layer_name].get("activation", None)
          if config_temp:
            if not CPUGPUQstrategy.config_equal(layer_name_config, config_temp):
              raise ValueError("Can not set different activation quant configs in the group {}".format(quant_group_name))
    
    return layer_name_config if layer_name_config else layer_activation_config

  @staticmethod
  def config_equal(config_a, config_b):
    config_equal_or_not = True
    for config_name, config_value in config_a.items():
      if config_value != config_b[config_name]:
        config_equal_or_not = False
    return config_equal_or_not
  
  
class TensorRTCGQStrategy(CPUGPUQstrategy):

  def create_quant_config(self, quant_info_mgr):
    config = {'param': {}, 'output': {}, 'input': {}}
    quant_algo = {'param': {}, 'output': {}, 'input': {}}
    
    for layer_type, layer_config in self._quant_strategy_info["layer_type_config"].items():
      nndct_layer_type = get_nndct_op_type(layer_type)
      self._layer_quant_types.append(nndct_layer_type)
    
    for node in quant_info_mgr.Nndctgraph.all_nodes():
      if quant_info_mgr.is_node_tensorrt_quantizable(node):
        # parameters
        for k in quant_info_mgr.quant_node_params(node).keys():
          p = quant_info_mgr.quant_node_params(node)[k]
          bw = self.num_bits_w
          # for mix precision quantization
          if (node.has_bound_params()):
            if (hasattr(node.op.ParamName, 'WEIGHTS') and k == node.op.ParamName.WEIGHTS):
              layer_weight_quant = self.get_layer_weights_config(node)
              if layer_weight_quant:
                bw = layer_weight_quant['bit_width']
                quant_algo['param'][p.name] = [create_quant_algo("weights", layer_weight_quant, node)]
              else:
                bw = self.num_bits_w
                quant_algo['param'][p.name] = [create_quant_algo("weights", self._quant_strategy_info["weights"], node)]
              # TODO: use set function set config
              config['param'][p.name] = [[bw, None, None, None]] # bitwidth, scale, zero_point
         
        # input blobs
        layer_activation_quant = self.get_layer_activation_config(node)
        config['input'][node.name] = []
        quant_algo['input'][node.name] = []
        for tensor in node.in_tensors:
          if tensor.name not in config['param'].keys():
            if layer_weight_quant:
              config['input'][node.name].append([layer_weight_quant['bit_width'], None, None, None])
              quant_algo['input'][node.name].append(create_quant_algo("weights", layer_weight_quant, node))
            else:
              config['input'][node.name].append([self.num_bits_w, None, None, None])
              quant_algo['input'][node.name].append(create_quant_algo("weights", self._quant_strategy_info["weights"], node))
        for tensor in node.in_tensors:
          if tensor.name not in config['param'].keys():
            if layer_activation_quant:
              config['input'][node.name].append([layer_activation_quant['bit_width'], None, None, None])
              quant_algo['input'][node.name].append(create_quant_algo("activation", layer_activation_quant, node))
            else:
              config['input'][node.name].append([self.num_bits_a, None, None, None])
              quant_algo['input'][node.name].append(create_quant_algo("activation", self._quant_strategy_info["activation"], node))
              
    return config, quant_algo

  def get_layer_activation_config(self, node):
    layer_activation_quant = None
    if node.op.type in self._layer_quant_types:
      torch_type = get_torch_op_type(node.op.type)
      layer_activation_quant = self._quant_strategy_info["layer_type_config"][torch_type].get("activation", None)
    return layer_activation_quant