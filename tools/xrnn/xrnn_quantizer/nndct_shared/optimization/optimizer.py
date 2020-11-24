

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

from nndct_shared.base import NNDCT_OP
from nndct_shared.nndct_graph import Graph
from .commander import OptimizeCommander
from nndct_shared.utils import NndctOption, NndctScreenLogger
import numpy as np

class QuantOptimizer(object):

  def __call__(self, graph: Graph, fuse_conv_bn: bool = True):
    
    commander = OptimizeCommander(graph=graph)
    if fuse_conv_bn:
      commander.FuseBnToConv()
      
      if NndctOption.nndct_equalization.value:
        NndctScreenLogger().info(f"=>Doing weights equalization...")
        #print("before equalization")
        #pre_cov = self._cal_channel_range_coeffience(graph)
        commander.equalize_weights_cross_conv_layers()
        #print("after equalization")
        #self._cal_channel_range_coeffience(graph, pre_cov)
      
    self._tag_quant_nodes(graph)
    graph.remove_node_by_types([NNDCT_OP.DROPOUT, NNDCT_OP.DEQUANT_STUB])
    # print(f"quant_state:")
    # for node in graph.nodes:
    #   print(f"{node.name}:{node.in_quant_part}")
    return graph
  
  @staticmethod
  def _tag_quant_nodes(graph: Graph):
    if any([node.op.type == NNDCT_OP.QUANT_STUB or node.op.type == NNDCT_OP.DEQUANT_STUB for node in graph.nodes]):
      quant_state = False
      for node in graph.nodes:
        if node.op.type == NNDCT_OP.QUANT_STUB:
          quant_state = True
          # for pn in graph.parents(node):
          #   if pn.op.type == NNDCT_OP.INPUT:
          #     pn.in_quant_part = quant_state
            
        elif node.op.type == NNDCT_OP.DEQUANT_STUB:
          quant_state = False
        node.in_quant_part = quant_state   
    else:
      for node in graph.nodes:
        node.in_quant_part = True
  
  '''
  @staticmethod     
  def _cal_channel_range_coeffience(graph, compared_cov=None):
    cov = {}
    for node in graph.nodes:
      if node.op.type in [NNDCT_OP.CONV2D, NNDCT_OP.CONVTRANSPOSE2D]:
        weight_tensor = node.op.params[node.op.ParamName.WEIGHTS]
        weight_max = np.max(np.fabs(weight_tensor.data))
        weight_std = np.std(np.fabs(weight_tensor.data))
        print("{name}: max:{max}, std:{std}".format(name=weight_tensor.name, max=weight_max, std=weight_std))
                  
        if node.node_attr(node.op.AttrName.BIAS_TERM):
          bias_tensor = node.op.params[node.op.ParamName.BIAS]
          bias_max = np.max(np.fabs(bias_tensor.data))
          bias_std = np.std(np.fabs(bias_tensor.data))
          print("{name}: max:{max}, std:{std}".format(name=bias_tensor.name, max=bias_max, std=bias_std))     
       
        range_per_c = np.max(np.fabs(weight_tensor.data), axis=(1, 2, 3))
        mean = range_per_c.mean()
        std = range_per_c.std()
        cov[weight_tensor.name] = std / mean
        if compared_cov is not None:
          print("{name}: after_cov = {after_cov}, \nbefore_cov = {before_cov}, \ndelta_cov = {delta_cov}".format(
            name=weight_tensor.name,
            after_cov=cov[weight_tensor.name],
            before_cov=compared_cov[weight_tensor.name],
            delta_cov=cov[weight_tensor.name] - compared_cov[weight_tensor.name]
          ))
      elif node.op.type == NNDCT_OP.DEPTHWISE_CONV2D:
        weight_tensor = node.op.params[node.op.ParamName.WEIGHTS]
        
        weight_max = np.max(np.fabs(weight_tensor.data))
        weight_std = np.std(np.fabs(weight_tensor.data))
        print("{name}: max:{max}, std:{std}".format(name=weight_tensor.name, max=weight_max, std=weight_std))

        if node.node_attr(node.op.AttrName.BIAS_TERM):
          bias_tensor = node.op.params[node.op.ParamName.BIAS]
          bias_max = np.max(np.fabs(bias_tensor.data))
          bias_std = np.std(np.fabs(bias_tensor.data))
          print("{name}: max:{max}, std:{std}".format(name=bias_tensor.name, max=bias_max, std=bias_std))
       
        range_per_c = np.max(np.fabs(weight_tensor.data), axis=(0, 1, 2))
        mean = range_per_c.mean()
        std = range_per_c.std()
        cov[weight_tensor.name] = std / mean
        if compared_cov is not None:
          print("{name}: after_cov = {after_cov}, \nbefore_cov = {before_cov}, \ndelta_cov = {delta_cov}".format(
            name=weight_tensor.name,
            after_cov=cov[weight_tensor.name],
            before_cov=compared_cov[weight_tensor.name],
            delta_cov=cov[weight_tensor.name] - compared_cov[weight_tensor.name]
          ))
      
    return cov
  '''
          
      
        
        
    




