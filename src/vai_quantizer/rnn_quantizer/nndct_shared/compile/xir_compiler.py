

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
from collections import namedtuple
from typing import Any, Dict, List, NoReturn, Optional

import numpy as np

from nndct_shared.base import NNDCT_OP
from nndct_shared.nndct_graph import Graph
from nndct_shared.nndct_graph import operator_definition as base_op
from nndct_shared.quantization import BaseQuantizer
from nndct_shared.utils import (AddXopError, NndctOption, GLOBAL_MAP, NNDCT_KEYS, NndctScreenLogger)

from .deploy_optimizer import DevGraphOptimizer
from .xgraph import XGraph
from .xop_creator import NNDCTIR2XIR_CONVERTOR, custom_xop, to_xir

NndctQuantInfo = Dict[str, Dict[str, List[int]]]

DeployGraphInfo = namedtuple("DeployGraphInfo", ["dev_graph", "quant_info"])
        
        
class XirCompiler(object):
  
  @staticmethod
  def get_xmodel_and_dump_infos(quantizer: BaseQuantizer, deploy_graphs_list: List[List[Graph]]):
    if len(deploy_graphs_list) == 1:
      graph_quant_info = XirCompiler.get_deloy_graph_infos(quantizer, deploy_graphs_list[0])
      return graph_quant_info, graph_quant_info
    elif len(deploy_graphs_list) == 2:
      xmodel_quant_info = XirCompiler.get_deloy_graph_infos(quantizer, deploy_graphs_list[0])
      dump_quant_info = XirCompiler.get_deloy_graph_infos(quantizer, deploy_graphs_list[1])
      return xmodel_quant_info, dump_quant_info
    else:
      raise RuntimeError(f"Length of graphs list to deploy should be 1 or 2")

  @staticmethod
  def get_deloy_graph_infos(quantizer: BaseQuantizer, deploy_graphs: List[Graph]) -> List[DeployGraphInfo]:
    graph_quant_info_list = []
    quant_groups = copy.deepcopy(quantizer.configer.quant_groups)
    quant_config = {"param": {}, "output": {}, "input": {}}
    if not NndctOption.nndct_quant_off.value:
      quant_config["param"].update(quantizer.quant_config["param"])
      quant_config["input"].update(quantizer.quant_config["input"])
      for blob_name, quant_info in quantizer.quant_config["output"].items():
        # if any([v is None for v in quant_info]):
        #   continue
        
        if any([blob_name in dev_graph for dev_graph in deploy_graphs]):
          quant_config["output"][blob_name] = copy.deepcopy(quant_info)
        else:
          if len(quant_groups[blob_name]) == 1:
            continue
  
          *prev_blobs, candidate_blob, blob_self = quant_groups[blob_name]
          if blob_self != blob_name:
            raise RuntimeError(f"Please check quant group:{blob_name}\n{quant_groups[blob_name]}")
          
          while prev_blobs:
            if all([candidate_blob not in dev_graph for dev_graph in deploy_graphs]):
              *prev_blobs, candidate_blob = prev_blobs
            else:
              break
            
          quant_config["output"][candidate_blob] = copy.deepcopy(quant_info)
        
    for dev_graph in deploy_graphs:
      graph_quant_info_list.append(DeployGraphInfo(dev_graph=dev_graph, quant_info=quant_config))
    
    return graph_quant_info_list
     
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
    
    #sorted_nodes = compile_graph.top_sort_nodeset(list(compile_graph.nodes))
    sorted_nodes = list(compile_graph.nodes)
    for node in sorted_nodes:
      for param_type, param_tensor in node.op.params.items():
        if (node.op.type in [NNDCT_OP.BATCH_NORM, NNDCT_OP.BATCH_NORM1D, NNDCT_OP.BATCH_NORM3D] 
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
          xgraph.create_fixed_const_op(
              name=param_tensor.name,
              data=data,
              quant_info=quant_config_info)
        except Exception as e:
          raise AddXopError(param_tensor.name, 'const', str(e))

    custom2xir = GLOBAL_MAP.get_ele(NNDCT_KEYS.CUSTOM_TO_XIR_LIST)
    if custom2xir:
      for op_type in custom2xir:
        NNDCTIR2XIR_CONVERTOR[op_type] = to_xir(op_type)
    
    for node in sorted_nodes:
      # print("convert...:", node.op.type, node.name, node.in_quant_part)
      # import sys
      # sys.stdout.flush()
      try:
        NNDCTIR2XIR_CONVERTOR.get(node.op.type, custom_xop)(xgraph, node, quant_config_info)
          
      except Exception as e:
        raise AddXopError(node.name, node.op.type, str(e))
    
      
    return_ops = []
    for tensor in compile_graph.end_tensors:
      op_name = xgraph.get_op_by_name(tensor.node.name).get_name()
      return_ops.append(op_name)
    if return_ops:
      xgraph.graph.set_attr("return_ops", return_ops)
      
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
    sorted_nodes = compile_graph.top_sort_nodeset(list(compile_graph.nodes))
    for node in sorted_nodes:
      if node.out_tensors[0].ndim and node.out_tensors[0].ndim > 1:
        xop_shape = xgraph.get_op_output_shape(node.name)
        if tuple(xop_shape) != tuple(node.out_tensors[0].shape):
          NndctScreenLogger().error(f"output shape of {node.name}({node.out_tensors[0].shape}) is different from the output shape of XIR ({xop_shape})")

        
                
    
