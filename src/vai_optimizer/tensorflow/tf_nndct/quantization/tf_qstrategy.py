
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

from typing import Dict, Union

from nndct_shared.base import NNDCT_OP
from nndct_shared.utils import NndctOption
from nndct_shared.quantization import QuantStrategyBase
from nndct_shared.quantization import QuantConfigImpBase

class TFLstmQstrategy(QuantStrategyBase):

  def __init__(self, quant_strategy_info: Dict[str, Union[str, int, bool]], 
               param_config: QuantConfigImpBase):
    super().__init__(quant_strategy_info, param_config, is_lstm=True)

  def create_quant_config(self, quant_info_mgr):
    return self._get_default_quant_config(quant_info_mgr, lstm=True)
  
  def _get_default_quant_config(self,
                                quant_info_mgr,
                                lstm=False):
    """
    1. unified activation bits
    2 .mixed bits for lstm 
    
    """
    print_log = (NndctOption.nndct_stat.value > 0)
    for node in quant_info_mgr.Nndctgraph.all_nodes():
      if print_log:
        print('---- Handling node %s type: %s' % (node.name, node.op.type))
      if quant_info_mgr.is_node_quantizable(node, lstm):
        # parameters
        for k in quant_info_mgr.quant_node_params(node).keys():
          p = quant_info_mgr.quant_node_params(node)[k]
          if p.name not in self._quant_config_imp.quant_config['param'].keys():
            # for mix precision quantization
            bw = self.num_bits_a
            if (node.has_bound_params() and 
              (hasattr(node.op.ParamName, 'WEIGHTS') and k == node.op.ParamName.WEIGHTS or
              hasattr(node.op.ParamName, 'GAMMA') and k == node.op.ParamName.GAMMA)):
              if (node.op.type is not NNDCT_OP.LAYER_NORM):
                bw = self.num_bits_w
            self._param_config.add_quant_config(p.name, bw, 'param')
            if print_log:
              print('---- Add fix of param %s' % p.name)
        # output blobs
        end = quant_info_mgr.quant_output(node.name).name
        if end not in self._param_config.quant_config['output']:
          for tensor in quant_info_mgr.quant_output(node.name).out_tensors:
            if tensor.name not in self._param_config.quant_config['param'].keys():
              self._param_config.add_quant_config(end, self.num_bits_a, 'output')
              if print_log:
                print('---- Add fix of output blob %s' % end)
        # input blobs (for mix precision quantization)
        if self.num_bits_w != self.num_bits_a:
          if node.op.type in [NNDCT_OP.DENSE, NNDCT_OP.CONV2D]:
            for tensor in node.in_tensors:
              if tensor.name not in self._param_config.quant_config['param'].keys():
                self._param_config.add_quant_config(node.name, self.num_bits_w, 'input')
                if print_log:
                  print('---- Add fix of input blob %s' % node.name)
      elif (lstm and (node in quant_info_mgr.Nndctgraph.inputs) and node.op.type not in [NNDCT_OP.BLOCK, NNDCT_OP.TUPLE_INPUT]):
        if print_log:
          print('---- Handling input node %s' % (node.name))
        # this path is only for quantizing a whole graph without quant stub OP
        # for lstm, check the following node type+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
        if (node.in_quant_part or (any(
            (quant_info_mgr.is_node_quantizable(c, lstm) and
             c.op.type is not NNDCT_OP.QUANT_STUB)
            for c in quant_info_mgr.Nndctgraph.children(node.name)))):
          end = quant_info_mgr.quant_output(node.name).name
          if end not in self._param_config.quant_configconfig['output']:
            for tensor in quant_info_mgr.quant_output(node.name).out_tensors:
              if tensor.name not in self._param_config.quant_config['param'].keys():
                self._param_config.add_quant_config(end, self.num_bits_a, 'output')
                if print_log:
                  print('---- Add fix of quant net input blob %s' % end)
    
    # check the input fix of all quantized ops 
    if not lstm:
      for node in quant_info_mgr.Nndctgraph.all_nodes():
        if quant_info_mgr.is_node_quantizable(node, lstm):
          if print_log:
            print('---- Check input of node %s type: %s' % (node.name, node.op.type))
          if node.op.type not in [NNDCT_OP.INPUT, NNDCT_OP.QUANT_STUB, NNDCT_OP.CONCAT]:
            for p_n in quant_info_mgr.Nndctgraph.parents(node):
              # if not quant_info_mgr.op_unquantizable(p_n.op.type):
                end = quant_info_mgr.quant_output(p_n.name).name
                end_node = quant_info_mgr.Nndctgraph.node(end)
                out_is_tensor = True
                for tensor in end_node.out_tensors:
                  if tensor.dtype not in ['tensor', 'float16', 'float32', 'float64']:
                    out_is_tensor = False
                if end not in self._param_config.quant_config['output'] and out_is_tensor:
                  for tensor in quant_info_mgr.quant_output(p_n.name).out_tensors:
                    if tensor.name not in self._param_config.quant_config['param'].keys():
                      self._param_config.add_quant_config(end, self.num_bits_a, 'output')
                      if print_log:
                        print('---- Add fix of output blob %s type: %s' % (end, end_node.op.type))
                  
          elif node.op.type in [NNDCT_OP.INPUT]:
            cn_nodes = quant_info_mgr.Nndctgraph.children(node)
            if len(cn_nodes) == 1 and cn_nodes[0].op.is_custom_op:
              end = quant_info_mgr.quant_output(node.name).name
              if end in self._param_config.quant_config['output']:
                del self._param_config.quant_config['output'][end]
                node.in_quant_part = False
    
    return self._param_config.quant_config, None
  