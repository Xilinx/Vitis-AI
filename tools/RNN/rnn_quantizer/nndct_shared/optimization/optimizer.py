

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
from .insert_node import NodeInsertHandler
import numpy as np

class QuantOptimizer(object):

  def __call__(self, graph: Graph, fuse_conv_bn: bool = True):
    
    commander = OptimizeCommander(graph=graph)
    commander.DecoupleSharedParamsInConv()
    if fuse_conv_bn:
      #commander.DecoupleSharedParamsInConv()
      commander.FuseBnToConv()
      commander.ConvertBNParams()
      
      if NndctOption.nndct_equalization.value:
        NndctScreenLogger().info(f"=>Doing weights equalization...")
        #print("before equalization")
        #pre_cov = self._cal_channel_range_coeffience(graph)
        graph = commander.equalize_weights_cross_conv_layers()
        #print("after equalization")
        #self._cal_channel_range_coeffience(graph, pre_cov)
      
      # if NndctOption.nndct_wes.value:
      #   NndctScreenLogger().info(f"=>Doing weights equalizing shift...")
      #   graph = commander.weights_equalizing_shift()
      
    if NndctOption.nndct_partition_mode.value > 0:
      self._tag_quant_nodes_v2(graph)
    else:
      self._tag_quant_nodes(graph)
    graph.remove_node_by_types([NNDCT_OP.DROPOUT])
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

  '''
  @staticmethod
  def _insert_scale_nodes(graph: Graph):
    conv_types = [NNDCT_OP.CONV2D, NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.CONVTRANSPOSE2D, \
                  NNDCT_OP.CONV3D, NNDCT_OP.CONVTRANSPOSE3D]
    conv_nodes = graph.find_nodes_by_types(conv_types)
    node_idx_max = np.array([node.idx for node in self._graph.nodes]).max()
    node_inserter = NodeInsertHandler(self._graph)
    #scale_idx = 0
    for conv_node in conv_nodes:
      
      #insert one channel scale node after conv node
      node_idx_max = node_idx_max + 1
      scale_node = node_inserter.insert_scale_node(conv_node, 1.0, node_idx_max)
      
      #insert another channel scale node 
      node_idx_max = node_idx_max + 1
      node_inserter.insert_scale_node(scale_node, 1.0, node_idx_max)
      
      conv_name = conv_node.name
      conv_node_out = conv_node.out_tensors[0]
      conv_shape = conv_node_out.shape
      
      scale1_op = TorchBaseOperation(NNDCT_OP.CHANNEL_SCALE, "Channel_Scale", force_to_primitive=True)
      scale1_op.set_config("channel_scale", 1.0)
      scale1_op.set_config("input", conv_node_out)
      
      node1_name = "/".join([conv_name, "channel_scale.1"])
      node1 = Node(node1_name, op=scale1_op, dtype="float32", idx=node_idx_max, in_quant_part=False)
      
      out_tensor1 = Tensor(name=f"{node_name}.0", node=node1, shape=conv_shape, dtype="float32", layout=conv_node_out.layout)
      node1.out_tensors.append(out_tensor1)
      
      node1.in_tensors.append(conv_node_out)
      
      out_nodes_name = conv_node.out_nodes
      for i in range(len(out_nodes_name)):
        out_node = self._graph.node(out_nodes_name[i])
        if (conv_node_out in out_node.in_tensors):
          node_index = out_node.in_tensors.index(conv_node_out)
          out_node.in_tensors[node_index] = out_tensor
        conv_node.out_nodes[i] = node.name
        in_node_idx = out_node.in_nodes.index(conv_node.name)
        out_node.in_nodes[in_node_idx] = node.name
        node.add_in_node(conv_node.name)
        node.add_out_node(out_node.name)

      self._graph.add_node(node)
  '''  
  
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
    
    else:
      for node in graph.nodes:
        node.in_quant_part = True
        
    # Debug   
    # for node in graph.nodes:
    #   print(node.name, node.op.type, node.in_quant_part)
  
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
                  
        if node.node_attr(node.op.AttrName.BIAS_TERM):
          bias_tensor = node.op.params[node.op.ParamName.BIAS]
          bias_max = np.max(np.fabs(bias_tensor.data))
          bias_std = np.std(np.fabs(bias_tensor.data))
          print("{name}: max:{max}, std:{std}".format(name=bias_tensor.name, max=bias_max, std=bias_std))
          
          bias_mean = bias_tensor.data.mean()
          bias_std = bias_tensor.data.std()
          cov[bias_tensor.name] = bias_std/bias_mean
          if compared_cov is not None:
            print("{name}: after_cov = {after_cov}, \nbefore_cov = {before_cov}, \ndelta_cov = {delta_cov}".format(
              name=bias_tensor.name,
              after_cov=cov[bias_tensor.name],
              before_cov=compared_cov[bias_tensor.name],
              delta_cov=cov[bias_tensor.name] - compared_cov[bias_tensor.name]
            ))
        
      elif node.op.type == NNDCT_OP.DEPTHWISE_CONV2D:
        weight_tensor = node.op.params[node.op.ParamName.WEIGHTS]
        
        weight_max = np.max(np.fabs(weight_tensor.data))
        weight_std = np.std(np.fabs(weight_tensor.data))
        print("{name}: max:{max}, std:{std}".format(name=weight_tensor.name, max=weight_max, std=weight_std))
        
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

        if node.node_attr(node.op.AttrName.BIAS_TERM):
          bias_tensor = node.op.params[node.op.ParamName.BIAS]
          bias_max = np.max(np.fabs(bias_tensor.data))
          bias_std = np.std(np.fabs(bias_tensor.data))
          print("{name}: max:{max}, std:{std}".format(name=bias_tensor.name, max=bias_max, std=bias_std))
          
          bias_mean = bias_tensor.data.mean()
          bias_std = bias_tensor.data.std()
          cov[bias_tensor.name] = bias_std/bias_mean
          if compared_cov is not None:
            print("{name}: after_cov = {after_cov}, \nbefore_cov = {before_cov}, \ndelta_cov = {delta_cov}".format(
              name=bias_tensor.name,
              after_cov=cov[bias_tensor.name],
              before_cov=compared_cov[bias_tensor.name],
              delta_cov=cov[bias_tensor.name] - compared_cov[bias_tensor.name]
            ))
          
    return cov
  '''
  
          
      
        
        
    




