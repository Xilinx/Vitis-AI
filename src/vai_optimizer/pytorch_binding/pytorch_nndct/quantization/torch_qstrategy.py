

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
from nndct_shared.utils import NndctScreenLogger, QError, QWarning
from nndct_shared.quantization import QuantStrategyBase
from nndct_shared.quantization import QuantConfigImpBase
from pytorch_nndct.quantization import create_quant_algo
from pytorch_nndct.utils.nndct2torch_op_map import get_nndct_op_type, get_torch_op_type

class TorchQstrategy(QuantStrategyBase):
  def __init__(self, quant_strategy_info: Dict[str, Union[str, int, bool]], 
               quant_config_imp: QuantConfigImpBase, is_lstm=False):
    super().__init__(quant_strategy_info, quant_config_imp, is_lstm=is_lstm)
    self._layer_quant_types = []
    self._layer_quant_names = []
    
  def _get_layer_quant_types_and_names(self):
    for layer_type, layer_config in self._quant_strategy_info["layer_type_config"].items():
      nndct_layer_type = get_nndct_op_type(layer_type)
      self._layer_quant_types.append(nndct_layer_type)
      
    for layer_name, layer_config in self._quant_strategy_info["layer_name_config"].items():
      self._layer_quant_names.append(layer_name)
  
  def get_layer_tensor_config(self, node, tensor_type='activation'):
    layer_tensor_quant = None
    if node.op.type in self._layer_quant_types:
      torch_type = get_torch_op_type(node.op.type)
      layer_tensor_quant = self._quant_strategy_info["layer_type_config"][torch_type].get(tensor_type, None)
    if node.name in self._layer_quant_names:
      layer_tensor_quant = self._quant_strategy_info["layer_name_config"][node.name].get(tensor_type, None)
    return self._quant_strategy_info[tensor_type] if layer_tensor_quant is None else layer_tensor_quant
  
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
            if not self.config_equal(layer_activation_config, config_temp):
              raise ValueError("Can not set different activation quant configs in the group {}".format(quant_group_name))
    
    for layer_name in self._layer_quant_names:
      if layer_name in quant_group_name:
        if layer_name_config == None:
          layer_name_config = self._quant_strategy_info["layer_name_config"][layer_name].get("activation", None)
        else:
          config_temp = self._quant_strategy_info["layer_name_config"][layer_name].get("activation", None)
          if config_temp:
            if not self.config_equal(layer_name_config, config_temp):
              raise ValueError("Can not set different activation quant configs in the group {}".format(quant_group_name))
    if (layer_name_config is None) and (layer_activation_config is None):
      return self._quant_strategy_info["activation"]
    else:
      return layer_name_config if layer_name_config else layer_activation_config
  
  @staticmethod
  def config_equal(config_a, config_b):
    config_equal_or_not = True
    for config_name, config_value in config_a.items():
      if config_value != config_b[config_name]:
        config_equal_or_not = False
    return config_equal_or_not
  
  @abstractmethod
  def create_quant_config(self, quant_info_mgr):
    pass
  
class TorchNndctQstrategy(TorchQstrategy):
  def create_quant_config(self, quant_info_mgr):
    self._quant_config_imp.clear_quant_config()
    
    print_log = (NndctOption.nndct_stat.value > 0)
    self._get_layer_quant_types_and_names()
    
    for node in quant_info_mgr.Nndctgraph.all_nodes():
      if print_log:
        print('---- Handling node %s type: %s' % (node.name, node.op.type))
      if quant_info_mgr.is_node_quantizable(node, self.lstm):
        # parameters
        for k in quant_info_mgr.quant_node_params(node).keys():
          p = quant_info_mgr.quant_node_params(node)[k]
          if p.name not in self._quant_config_imp.quant_config['param'].keys():
            quant_config = self._quant_strategy_info["activation"]
            # for mix precision quantization
            if (node.has_bound_params()):
              if (hasattr(node.op.ParamName, 'WEIGHTS') and k == node.op.ParamName.WEIGHTS or
              hasattr(node.op.ParamName, 'GAMMA') and k == node.op.ParamName.GAMMA):
                if (node.op.type is not NNDCT_OP.LAYER_NORM):
                  quant_config = self.get_layer_tensor_config(node, 'weights')
                self._quant_config_imp.add_quant_algo(p.name, create_quant_algo(quant_config, node), 'param')
              elif (hasattr(node.op.ParamName, 'BIAS') and k == node.op.ParamName.BIAS or
              hasattr(node.op.ParamName, 'BETA') and k == node.op.ParamName.BETA):
                quant_config = self.get_layer_tensor_config(node, 'bias')
                self._quant_config_imp.add_quant_algo(p.name, create_quant_algo(quant_config, node), 'param')
            # TODO: use set function set config
            self._quant_config_imp.add_quant_config(p.name, quant_config['bit_width'], 'param')
            if print_log:
              print('---- Add fix of param %s' % p.name)
        
        # output blobs
        end = quant_info_mgr.quant_output(node.name).name
        if end not in self._quant_config_imp.quant_config['output']:
          layer_group_quant = self.get_quant_group_activation_config(quant_info_mgr, node.name)
          for tensor in quant_info_mgr.quant_output(node.name).out_tensors:
            if tensor.name not in self._quant_config_imp.quant_config['param'].keys():
              self._quant_config_imp.add_quant_config(end, layer_group_quant['bit_width'], 'output')
              self._quant_config_imp.add_quant_algo(end, create_quant_algo(layer_group_quant, node), 'output')
              if print_log:
                print('---- Add fix of output blob %s' % end)
                
        # input blobs (for mix precision quantization)
        if node.op.type in [NNDCT_OP.DENSE, NNDCT_OP.CONV2D]:
          layer_weight_quant = self.get_layer_tensor_config(node, 'weights')
          layer_input_quant = self.get_layer_tensor_config(node, 'input')
          if layer_weight_quant['bit_width'] != layer_input_quant['bit_width']:
            layer_input_quant['bit_width'] = layer_weight_quant['bit_width']
            for tensor in node.in_tensors:
              if tensor.name not in self._quant_config_imp.quant_config['param'].keys():
                self._quant_config_imp.add_quant_config(node.name, layer_input_quant['bit_width'], 'input')
                self._quant_config_imp.add_quant_algo(node.name, create_quant_algo(layer_input_quant, node), 'input')
                if print_log:
                  print('---- Add fix of input blob %s' % node.name)
      
      elif (self.lstm and (node in quant_info_mgr.Nndctgraph.inputs) and node.op.type not in [NNDCT_OP.BLOCK, NNDCT_OP.TUPLE_INPUT]):
        # this path is only for quantizing a whole graph without quant stub OP
        # for lstm, check the following node type
        if print_log:
          print('---- Handling input node %s' % (node.name))
        if (node.in_quant_part or (any(
            (quant_info_mgr.is_node_quantizable(c, self.lstm) and
             c.op.type is not NNDCT_OP.QUANT_STUB)
            for c in quant_info_mgr.Nndctgraph.children(node.name)))):
          end = quant_info_mgr.quant_output(node.name).name
          if end not in self._quant_config_imp.quant_config['output']:
            layer_group_quant = self.get_quant_group_activation_config(quant_info_mgr, node.name)
            self._quant_config_imp.quant_config['output'][end] = []
            for tensor in quant_info_mgr.quant_output(node.name).out_tensors:
              if tensor.name not in self._quant_config_imp.quant_config['param'].keys():
                #config['output'][end].append([self.num_bits_a, None])
                self._quant_config_imp.add_quant_config(end, layer_group_quant['bit_width'], 'output')
                self._quant_config_imp.add_quant_algo(end, create_quant_algo(layer_group_quant, node), 'output')
                if print_log:
                  print('---- Add fix of quant net input blob %s' % end)
    
    # check the input fix of all quantized ops 
    if not self.lstm:
      for node in quant_info_mgr.Nndctgraph.all_nodes():
        if quant_info_mgr.is_node_quantizable(node, self.lstm):
          if print_log:
            print('---- Check input of node %s type: %s' % (node.name, node.op.type))
          if node.op.type not in [NNDCT_OP.INPUT, NNDCT_OP.QUANT_STUB, NNDCT_OP.CONCAT]:
            for p_n in quant_info_mgr.Nndctgraph.parents(node):
              # if not quant_info_mgr.op_unquantizable(p_n.op.type):
                end = quant_info_mgr.quant_output(p_n.name).name
                end_node = quant_info_mgr.Nndctgraph.node(end)
                out_is_tensor = False
                for tensor in end_node.out_tensors:
                  if tensor.dtype in ['tensor', 'float16', 'float32', 'float64']:
                    out_is_tensor = True
                
                if end not in self._quant_config_imp.quant_config['output'] and out_is_tensor:
                  layer_group_quant = self.get_quant_group_activation_config(quant_info_mgr, p_n.name)
                  for tensor in quant_info_mgr.quant_output(p_n.name).out_tensors:
                    if (tensor.name not in self._quant_config_imp.quant_config['param'].keys()) and \
                    (tensor.dtype in ['tensor', 'float16', 'float32', 'float64']):
                      self._quant_config_imp.add_quant_config(end, layer_group_quant['bit_width'], 'output')
                      self._quant_config_imp.add_quant_algo(end, create_quant_algo(layer_group_quant, node), 'output')
                      if print_log:
                        print('---- Add fix of output blob %s type: %s' % (end, end_node.op.type))
                  
          elif node.op.type in [NNDCT_OP.INPUT]:
            cn_nodes = quant_info_mgr.Nndctgraph.children(node)
            if len(cn_nodes) == 1 and cn_nodes[0].op.is_custom_op:
              end = quant_info_mgr.quant_output(node.name).name
              if end in self._quant_config_imp.quant_config['output']:
                del self._quant_config_imp.quant_config['output'][end]
                node.in_quant_part = False  
              
    return self._quant_config_imp.quant_config, self._quant_config_imp.quant_algo

class TorchMPQstrategy(TorchQstrategy):
  
  def _make_mix_precision_quant_config(self, name, node, config, quant_info_mgr, tensor_type):
    if config['datatype'] == 'int':
      if quant_info_mgr.is_node_quantizable(node, self.lstm):
        self._quant_config_imp.add_quant_config(name, config['bit_width'], tensor_type)
        self._quant_config_imp.add_quant_algo(name, create_quant_algo(config, node), tensor_type)
        self._quant_config_imp.add_quant_dtype(name, config['datatype'], tensor_type)
      else:
        NndctScreenLogger().warning_once(f"{node.op.type} is not supported in int quantization, skip it.")
    else:
      self._quant_config_imp.add_quant_dtype(name, config['datatype'], tensor_type)
      
  def create_quant_config(self, quant_info_mgr):
    self._quant_config_imp.clear_quant_config()
    
    print_log = (NndctOption.nndct_stat.value > 0)
    self._get_layer_quant_types_and_names()
    
    for node in quant_info_mgr.Nndctgraph.all_nodes():
      if print_log:
        print('---- Handling node %s type: %s' % (node.name, node.op.type))
      if node.op.type in [NNDCT_OP.INPUT, NNDCT_OP.TUPLE_INPUT]:
        if print_log:
          print('---- Skip node %s quantization type: %s' % (node.name, node.op.type))
        continue
      # parameters
      for k in quant_info_mgr.quant_node_params(node).keys():
        p = quant_info_mgr.quant_node_params(node)[k]
        if p.name not in self._quant_config_imp.quant_dtype['param'].keys():
          if (node.has_bound_params()):
            if (hasattr(node.op.ParamName, 'WEIGHTS') and k == node.op.ParamName.WEIGHTS or
              hasattr(node.op.ParamName, 'GAMMA') and k == node.op.ParamName.GAMMA):
              quant_config = self.get_layer_tensor_config(node, 'weights')
              self._make_mix_precision_quant_config(p.name, node, quant_config, quant_info_mgr, 'param')
            elif (hasattr(node.op.ParamName, 'BIAS') and k == node.op.ParamName.BIAS or
              hasattr(node.op.ParamName, 'BETA') and k == node.op.ParamName.BETA):
              quant_config = self.get_layer_tensor_config(node, 'bias')
              self._make_mix_precision_quant_config(p.name, node, quant_config, quant_info_mgr, 'param')
          # TODO: use set function set config
          if print_log:
            print('---- Add fix of param %s' % p.name)
        
      # output blobs
      end = quant_info_mgr.quant_output(node.name)
      if end.name not in self._quant_config_imp.quant_dtype['output']:
        layer_output_quant = self.get_layer_tensor_config(end, 'activation')
        for tensor in quant_info_mgr.quant_output(node.name).out_tensors:
          if tensor.name not in self._quant_config_imp.quant_dtype['param'].keys():
            self._make_mix_precision_quant_config(end.name, node, layer_output_quant, quant_info_mgr, 'output')
            if print_log:
              print('---- Add fix of output blob %s' % end.name)
              
      # input blobs
      layer_input_quant = self.get_layer_tensor_config(node, 'input')
      for tensor in node.in_tensors:
        if tensor.name not in self._quant_config_imp.quant_dtype['param'].keys():
          self._make_mix_precision_quant_config(node.name, node, layer_input_quant, quant_info_mgr, 'input')

    return self._quant_config_imp.quant_config, self._quant_config_imp.quant_algo
  
class TorchGemm88Qstrategy(TorchQstrategy):
  def create_quant_config(self, quant_info_mgr):
    """
    1. unified activation bits
    2 .mixed bits for lstm 
    
    """
    self._quant_config_imp.clear_quant_config()
    
    self._get_layer_quant_types_and_names()
    print_log = (NndctOption.nndct_stat.value > 0)
    gemm88_quant_op_list = [NNDCT_OP.DENSE,NNDCT_OP.MATMUL]
    for node in quant_info_mgr.Nndctgraph.all_nodes():
      if print_log:
        print('---- Handling node %s type: %s' % (node.name, node.op.type))
      if node.op.type in gemm88_quant_op_list:
        quant_config = self.get_layer_tensor_config(node, 'weights')
        if quant_info_mgr.is_node_quantizable(node, self.lstm):
          # parameters
          for k in quant_info_mgr.quant_node_params(node).keys():
            p = quant_info_mgr.quant_node_params(node)[k]
            if p.name not in self._quant_config_imp.quant_config['param'].keys():
              if hasattr(node.op.ParamName, 'WEIGHTS') and k == node.op.ParamName.WEIGHTS:
                self._quant_config_imp.add_quant_config(p.name, quant_config['bit_width'], 'param')
              if print_log:
                print('---- Add fix of param %s' % p.name)
          
          for tensor in node.in_tensors:
            if tensor.name not in self._quant_config_imp.quant_config['param'].keys():
              if tensor.name.split(".")[-1] != "bias":
                self._quant_config_imp.add_quant_config(node.name, quant_config['bit_width'], 'input')
                if print_log:
                  print('---- Add fix of input blob %s' % node.name)

    return self._quant_config_imp.quant_config, None

class TorchTRTQstrategy(TorchQstrategy):
  def create_quant_config(self, quant_info_mgr):
    self._quant_config_imp.clear_quant_config()
    self._get_layer_quant_types_and_names()
    
    for node in quant_info_mgr.Nndctgraph.all_nodes():
      if quant_info_mgr.is_node_quantizable(node, self.lstm):
        # parameters
        for k in quant_info_mgr.quant_node_params(node).keys():
          p = quant_info_mgr.quant_node_params(node)[k]
          if p.name not in self._quant_config_imp.quant_config['param'].keys():
            bw = self.num_bits_w
            # for mix precision quantization
            if (node.has_bound_params()):
              if (hasattr(node.op.ParamName, 'WEIGHTS') and k == node.op.ParamName.WEIGHTS):
                layer_weight_quant = self.get_layer_tensor_config(node, 'weights')
                self._quant_config_imp.add_quant_algo(p.name, create_quant_algo(layer_weight_quant, node), 'param')
                self._quant_config_imp.add_quant_config(p.name, layer_weight_quant['bit_width'], 'param')
         
        # input blobs
        layer_input_quant = self.get_layer_tensor_config(node, 'input')
        for tensor in node.in_tensors:
          if tensor.name not in self._quant_config_imp.quant_config['param'].keys():
            self._quant_config_imp.add_quant_config(node.name, layer_input_quant['bit_width'], 'input')
            self._quant_config_imp.add_quant_algo(node.name, create_quant_algo(layer_input_quant, node), 'input')

    return self._quant_config_imp.quant_config, self._quant_config_imp.quant_algo

