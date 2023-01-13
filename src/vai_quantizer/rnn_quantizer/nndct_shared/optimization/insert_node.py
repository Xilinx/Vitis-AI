
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

import numpy as np
from nndct_shared.base import NNDCT_OP
from nndct_shared.nndct_graph import Graph, Node, Tensor
from nndct_shared.nndct_graph import reorder_multi_subgraph_nodes
from pytorch_nndct.parse.torch_op_def import *

_INSERT_TYPES = [NNDCT_OP.CONV2D, NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.CONVTRANSPOSE2D, \
                 NNDCT_OP.CONV3D, NNDCT_OP.DEPTHWISE_CONV3D, NNDCT_OP.CONVTRANSPOSE3D, \
                 NNDCT_OP.CHANNEL_SCALE]

class NodeInsertHandler(object):
  def __init__(self, graph):
    self._graph = graph
    
  def insert_scale_node(self, node, scale_param, node_idx):
    if not node.op.type in _INSERT_TYPES:
      raise ValueError('Scale node only can insert after conv or scale node')
    
    scale_node = self._generate_scale_node(node, scale_param, node_idx)
    
    out_nodes_name = node.out_nodes
    if len(out_nodes_name) > 0:
      for i in range(len(out_nodes_name)):
        out_node = self._graph.node(out_nodes_name[i])
        
        if (scale_node.in_tensors[0] in out_node.in_tensors):
          tensor_index = out_node.in_tensors.index(scale_node.in_tensors[0])
          out_node.in_tensors[tensor_index] = scale_node.out_tensors[0]
          
        node.out_nodes[i] = scale_node.name
        
        in_node_idx = out_node.in_nodes.index(node.name)
        out_node.in_nodes[in_node_idx] = scale_node.name
        
        scale_node.add_in_node(node.name)
        scale_node.add_out_node(out_node.name)
    else:
      node.add_out_node(scale_node.name)
      scale_node.add_in_node(node.name)
      
    self._graph.add_node(scale_node)
    return scale_node
    
  def refactor_graph(self):
    end_tensors = self._graph.end_tensors
    copy_tensor_map = self._graph.copy_tensor_map
    self._graph = self._graph.create_subgraph_from_nodeset(self._graph, list(self._graph.nodes), self._graph.name)
    reorder_multi_subgraph_nodes([self._graph])
    self._graph.end_tensors = end_tensors
    self._graph.copy_tensor_map = copy_tensor_map
    return self._graph
    
    
  @staticmethod
  def _generate_scale_node(node, scale_param, node_idx):
    
    node_name = node.name
    tensor_out = node.out_tensors[0]
    out_tensor_shape = tensor_out.shape
    
    scale_op = TorchBaseOperation(NNDCT_OP.CHANNEL_SCALE, "Channel_Scale", force_to_primitive=True)
    scale_op.set_config("channel_scale", scale_param)
    scale_op.set_config("input", tensor_out)
    
    if node.op.type == NNDCT_OP.CHANNEL_SCALE:
      scale_node_name = node_name[:node_name.rfind(".")]+".2"
    else:
      scale_node_name = "/".join([node_name, "channel_scale.1"])
    
    scale_node = Node(scale_node_name, op=scale_op, dtype="float32", idx=node_idx, in_quant_part=False)
    
    out_tensor = Tensor(name=f"{scale_node_name}.0", node=scale_node, shape=out_tensor_shape, dtype="float32", layout=tensor_out.layout)
    scale_node.out_tensors.append(out_tensor)
    
    scale_node.in_tensors.append(tensor_out)
    return scale_node
  

