from os import environ
from typing import Any, Dict, List, NoReturn, Optional

import numpy as np

from nndct_shared.base import NNDCT_KEYS, NNDCT_OP
from nndct_shared.nndct_graph import Graph, Tensor
from nndct_shared.optimization import GraphOptimizer
from nndct_shared.utils import (AddXopError, ExportXmodelError, NndctOption,
                                NndctScreenLogger)

from .xgraph import XGraph
from .xop_creator import NNDCTIR2XIR_CONVERTOR, unsupported_xop

NndctQuantInfo = Dict[str, Dict[str, List[int]]]


class XirCompiler(object):
  
  @staticmethod
  def do_compile(nndct_graph: Graph,
                 output_file_name=None,
                 quant_config_info: Optional[NndctQuantInfo] = None,
                 graph_attr_kwargs: Optional[Dict[str, Any]] = None,
                ) -> NoReturn:
    
    r""" convert nndct graph to xmodel"""
    # debug
    # for type, bnfp in quant_config_info.items():
    #   print(f"{type}\n")
    #   for name, bnfp_value in bnfp.items():
    #     print(f"{name}:{bnfp_value}\n")

    if NndctOption.nndct_quant_off.value:
      quant_config_info = None
    
    xoptmizer = GraphOptimizer(nndct_graph)
    frozen_graph = xoptmizer.get_frozen_graph()
    
    xgraph = XGraph(frozen_graph.name)
    
    if graph_attr_kwargs is not None:
      for name, attr in graph_attr_kwargs.items():
        xgraph.graph.set_attr(name, attr)
        
    for node in frozen_graph.nodes:
      for param_type, param_tensor in node.op.params.items():
        data = np.copy(param_tensor.data)
        if node.op.type == NNDCT_OP.CONVTRANSPOSE2D and param_type == node.op.ParamName.WEIGHTS:
          # OHWI -> OH'W'I reverse the order of ele in both h and w axis
          data = np.flip(data, (1, 2))
          data = np.ascontiguousarray(data)
        try:
          xgraph.create_fixed_const_op(
              name=param_tensor.name,
              data=data,
              quant_info=quant_config_info)
        except Exception as e:
          raise AddXopError(param_tensor.name, 'const', str(e))

    for node in frozen_graph.nodes:
      try:
        NNDCTIR2XIR_CONVERTOR.get(node.op.type, unsupported_xop)(xgraph, node,
                                                              quant_config_info)
      except Exception as e:
        raise AddXopError(node.name, node.op.type, str(e))
        # print(f"{node.name}, {node.op.type}, {str(e)}")
    
    return_ops = []
    for tensor in frozen_graph.get_end_tensors():
      op_name = xgraph.get_op_by_name(tensor.node.name).get_name()
      return_ops.append(op_name)
    xgraph.graph.set_attr("return_ops", return_ops)
      
    if output_file_name:
      if quant_config_info is None:
        output_file_name += '_float'
      else:
        output_file_name += '_int'
        
      XirCompiler.xmodel_file = output_file_name + NNDCT_KEYS.XMODEL_SUFFIX
      
      try:
        xgraph.export_to_xmodel(XirCompiler.xmodel_file)
      except Exception:
        raise ExportXmodelError(frozen_graph.name)
      try:
        xgraph.export_to_img(output_file_name + NNDCT_KEYS.XMODEL_IMAGE_SUFFIX)
      except Exception as e:
        NndctScreenLogger().warning(f"Failed to generate xmodel image!({str(e)})")
