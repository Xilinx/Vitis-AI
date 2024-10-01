

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
    commander.DecoupleSharedParamsInConv()
    if NndctOption.nndct_ip_asr.value is True:
      commander.FuseEmbedLnActv()
    if fuse_conv_bn:
      commander.FuseBnToConv()
      commander.ConvertBNParams()
      
    graph.remove_node_by_types([NNDCT_OP.DROPOUT])
      
    if NndctOption.nndct_equalization.value:
      NndctScreenLogger().info(f"=>Doing weights equalization...")
      graph = commander.equalize_weights_cross_conv_layers()

    if NndctOption.nndct_partition_mode.value > 0:
      self._tag_quant_nodes_v2(graph)
    else:
      self._tag_quant_nodes(graph)

    #graph.remove_node_by_types([NNDCT_OP.DROPOUT])

    if NndctOption.nndct_leaky_relu_approximate.value:
      commander.SetNegativeSlope()
    # print(f"quant_state:")
    # for node in graph.nodes:
    #   print(f"{node.name}:{node.in_quant_part}")
    return graph
  
  @staticmethod
  def _tag_quant_nodes_v2(graph):
    def dfs(node, visited):
      if node.op.type != NNDCT_OP.INPUT:
        if not node.op.is_custom_op:
          node.in_quant_part = True
        else: 
          node.in_quant_part = False
      else:
        has_non_custom_op_child = False
        for cn in graph.children(node):
          if not cn.op.is_custom_op:
            has_non_custom_op_child = True
            break
        if has_non_custom_op_child:
          node.in_quant_part = True
      
      visited.append(node)
      for cn in graph.children(node):
        if cn not in visited:
          dfs(cn, visited)
    
    # initialize the flag
    for node in graph.nodes:
      node.in_quant_part = False
        
    visited = []
    for nd in graph.nodes:
      dfs(nd, visited)
    
    # Debug   
    #for node in graph.nodes:
    #  print(node.name, node.op.type, node.in_quant_part)
  
  @staticmethod
  def _tag_quant_nodes(graph):
    if any([node.op.type == NNDCT_OP.QUANT_STUB or node.op.type == NNDCT_OP.DEQUANT_STUB for node in graph.nodes]):
      def dfs(node, visited):
        if node.op.type == NNDCT_OP.DEQUANT_STUB:
          node.in_quant_part = False
          return
        
        visited.append(node)
        node.in_quant_part = True
        for cn in graph.children(node):
          if cn not in visited:
            dfs(cn, visited)
      
      source = []
      for node in graph.nodes:
        if node.op.type == NNDCT_OP.QUANT_STUB:
          source.append(node)
          
      visited = []
      for inp in source:
        dfs(inp, visited)

      for node in graph.nodes:
        if not node.in_nodes and node.op.type != NNDCT_OP.INPUT:
          for cn in graph.children(node):
            if cn.op.type != NNDCT_OP.QUANT_STUB and cn.in_quant_part:
              node.in_quant_part = True

    else:
      for node in graph.all_nodes():
        node.in_quant_part = True
        
    # Debug   
    # print("\n quant part")
    # for node in graph.nodes:
    #   print(node.name, node.op.type, node.in_quant_part)
  
 
      
        
        
    




