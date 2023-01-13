

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

from typing import Any, Dict, List, NoReturn, Optional

import numpy as np

from nndct_shared.base import NNDCT_OP
from nndct_shared.nndct_graph import Graph
from nndct_shared.utils import (AddXopError, NndctOption, GLOBAL_MAP, NNDCT_KEYS, NndctScreenLogger)
from nndct_shared.utils import QError, QWarning
from .xgraph import XGraph
from .xop_creator import NNDCTIR2XIR_CONVERTOR, custom_xop, to_xir

NndctQuantInfo = Dict[str, Dict[str, List[int]]]
     
     
class XirCompiler(object): 
  @staticmethod
  def do_compile(compile_graph: Graph,
                 output_file_name=None,
                 quant_config_info: Optional[NndctQuantInfo] = None,
                 graph_attr_kwargs: Optional[Dict[str, Any]] = None) -> NoReturn:
    
    r""" convert nndct graph to xmodel"""
    # debug
    # for type, bnfp in quant_config_info.items():
    #   print(f"{type}\n")
    #   for name, bnfp_value in bnfp.items():
    #     print(f"{name}:{bnfp_value}\n")
    if NndctOption.nndct_quant_off.value:
      quant_config_info = None
    
    xgraph = XGraph(compile_graph.name)
    
    if graph_attr_kwargs is not None:
      for name, attr in graph_attr_kwargs.items():
        xgraph.graph.set_attr(name, attr)
    
    for node in compile_graph.nodes:
      for param_type, param_tensor in node.op.params.items():
        if (node.op.type == NNDCT_OP.BATCH_NORM 
            and param_type not in [node.op.ParamName.GAMMA, node.op.ParamName.BETA]):
          continue
        if xgraph.get_op_by_name(param_tensor.name):
          continue
        # print(f"{node.name}: {param_tensor.name}, {id(param_tensor)}")
        data = np.copy(param_tensor.data)
        if node.op.type in [NNDCT_OP.CONVTRANSPOSE2D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D] and param_type == node.op.ParamName.WEIGHTS:
          # OHWI -> OH'W'I reverse the order of ele in both h and w axis
          data = np.flip(data, (1, 2))
          data = np.ascontiguousarray(data)
        elif node.op.type in [NNDCT_OP.CONVTRANSPOSE3D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D] and param_type == node.op.ParamName.WEIGHTS:
           # OHWDI -> OH'W'D'I reverse the order of ele in both h and w axis
          data = np.flip(data, (1, 2, 3))
          data = np.ascontiguousarray(data)
        try:
          if data.dtype == np.float16:
            data = data.astype(np.float32)
          xgraph.create_fixed_const_op(
              name=param_tensor.name,
              data=data,
              quant_info=quant_config_info)
        except Exception as e:
          raise AddXopError(param_tensor.name, 'const', str(e))

    custom2xir = GLOBAL_MAP.get_ele(NNDCT_KEYS.CUSTOM_TO_XIR_LIST)
    if custom2xir:
      for op_type in custom2xir:
        NNDCTIR2XIR_CONVERTOR[op_type] = (op_type, to_xir(op_type))
   
    for node in compile_graph.nodes:
      if node.op.type == NNDCT_OP.RETURN:
          continue
      # print("convert...:", node.op.type, node.name, node.in_quant_part)
      # import sys
      # sys.stdout.flush()
      try:
        NNDCTIR2XIR_CONVERTOR.get(node.op.type, (node.op.type, custom_xop))[1](xgraph, node, quant_config_info)
      except Exception as e:
        raise AddXopError(node.name, node.op.type, str(e))
      
    if output_file_name:
      if quant_config_info is None:
        output_file_name += '_float'
      else:
        output_file_name += '_int'
              
      xgraph.export_to_xmodel(output_file_name)
      
    return xgraph

  @staticmethod
  def verify_xmodel(compile_graph: Graph, xgraph: XGraph):
    """verify the xmodel by nndct node shape"""
   
    for node in compile_graph.nodes:
      if not node.out_tensors:
          continue
      if node.out_tensors[0].ndim and node.out_tensors[0].ndim > 1:
        xop_shape = xgraph.get_op_output_shape(node.name)
        if tuple(xop_shape) != tuple(node.out_tensors[0].shape):
          NndctScreenLogger().error2user(QError.SHAPE_MISMATCH, f"output shape of {node.name}({node.out_tensors[0].shape}) is different from the output shape of XIR ({xop_shape}).")

        
                
    
  @staticmethod
  def verify_nndct_graph(compile_graph):
    msg = ""
    for node in compile_graph.nodes:
      if node.op.type == NNDCT_OP.RETURN:
        continue
      if node.blocks:
        msg += f"XIR don't support control flow op.({node.name}, {node.op.type})\n"
      elif len(node.out_tensors) > 1 and all([len(tensor.uses) > 0 for tensor in node.out_tensors]):
        msg += f"XIR don't support multi-outputs op.({node.name}, {node.op.type})\n"
      elif node.op.type not in NNDCTIR2XIR_CONVERTOR.keys() and all([tensor.shape is None for tensor in node.out_tensors]):
        msg += f"XIR don't support custom op without shape info.({node.name}, {node.op.type})\n"
      
    if msg:
      return False, msg
  
    return True, msg

