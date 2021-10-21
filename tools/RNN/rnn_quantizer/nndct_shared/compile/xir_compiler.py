

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
from nndct_shared.quantization import BaseQuantizer
from nndct_shared.utils import (AddXopError, NndctOption)

from .deploy_optimizer import DevGraphOptimizer
from .xgraph import XGraph
from .xop_creator import NNDCTIR2XIR_CONVERTOR, unsupported_xop

NndctQuantInfo = Dict[str, Dict[str, List[int]]]

DeployGraphInfo = namedtuple("DeployGraphInfo", ["dev_graph", "quant_info"])
        
        
class XirCompiler(object):
  
  @staticmethod
  def get_deloy_graph_infos(quantizer: BaseQuantizer, deploy_graphs: List[Graph]) -> List[DeployGraphInfo]:
    # g_optmizer = DevGraphOptimizer(copied_nndct_graph)
    # g_optmizer.freeze_graph()
    # deploy_graphs = g_optmizer.partition_by_quant_part()
    graph_quant_info_list = []
    quant_groups = copy.deepcopy(quantizer.configer.quant_groups)
    # for dev_graph in deploy_graphs:
    quant_config = {"param": {}, "output": {}, "input": {}}
    if not NndctOption.nndct_quant_off.value:
      quant_config["param"].update(quantizer.quant_config["param"])
      quant_config["input"].update(quantizer.quant_config["input"])
      for blob_name, quant_info in quantizer.quant_config["output"].items():
        if any([v is None for v in quant_info]):
          continue
        
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
        
    for node in compile_graph.nodes:
      for param_type, param_tensor in node.op.params.items():
        if (node.op.type in [NNDCT_OP.BATCH_NORM, NNDCT_OP.BATCH_NORM1D, NNDCT_OP.BATCH_NORM3D] 
            and param_type not in [node.op.ParamName.GAMMA, node.op.ParamName.BETA]):
          continue
        if xgraph.get_op_by_name(param_tensor.name):
          continue
        # print(f"{node.name}: {param_tensor.name}, {id(param_tensor)}")
        data = np.copy(param_tensor.data)
        if node.op.type == NNDCT_OP.CONVTRANSPOSE2D and param_type == node.op.ParamName.WEIGHTS:
          # OHWI -> OH'W'I reverse the order of ele in both h and w axis
          data = np.flip(data, (1, 2))
          data = np.ascontiguousarray(data)
        elif node.op.type == NNDCT_OP.CONV3D and param_type == node.op.ParamName.WEIGHTS:
          data = data.transpose(0, 2, 3, 4, 1)
          data = np.ascontiguousarray(data)
        elif node.op.type == NNDCT_OP.CONVTRANSPOSE3D and param_type == node.op.ParamName.WEIGHTS:
          data = data.transpose(1, 2, 3, 4, 0)
          data = np.ascontiguousarray(data)
        try:
          xgraph.create_fixed_const_op(
              name=param_tensor.name,
              data=data,
              quant_info=quant_config_info)
        except Exception as e:
          raise AddXopError(param_tensor.name, 'const', str(e))

    unknown_op_types = {f"{node.op.type}({node.name})" for node in compile_graph.nodes 
                        if node.op.type not in NNDCTIR2XIR_CONVERTOR}
    if not unknown_op_types:
      for node in compile_graph.nodes:
        try:
          NNDCTIR2XIR_CONVERTOR.get(node.op.type, unsupported_xop)(xgraph, node, quant_config_info)
        except Exception as e:
          raise AddXopError(node.name, node.op.type, str(e))
    else:
      # NndctScreenLogger().error(f"Please support these ops in XIR:{unknown_op_types}.")
      raise AddXopError(unknown_op_types) 
      
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
      # xgraph.export_to_img(output_file_name)       
