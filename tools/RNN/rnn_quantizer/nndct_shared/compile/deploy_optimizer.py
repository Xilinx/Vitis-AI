

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
from nndct_shared.base import NNDCT_OP
from nndct_shared.nndct_graph import Tensor, Graph, Node
from nndct_shared.utils import NndctDebugLogger, NndctOption
from typing import List, Dict
from .op_evaluator import Evaluator
from collections import defaultdict


class DevGraphOptimizer(object):
  """Optimze graph for device computation
 
  """

  def __init__(self, nndct_graph):
    self._dev_graph = copy.deepcopy(nndct_graph)
    self._evalute_func_map = {
        NNDCT_OP.SHAPE: Evaluator.shape,
        NNDCT_OP.CAST: Evaluator.cast,
        NNDCT_OP.INT: Evaluator.int,
        NNDCT_OP.SCALAR_MUL: Evaluator.mul,
        NNDCT_OP.TENSOR: Evaluator.tensor,
        NNDCT_OP.FLOOR: Evaluator.floor,
        NNDCT_OP.DIV: Evaluator.elemwise_div,
        NNDCT_OP.FLOOR_DIV: Evaluator.floor_div, 
        NNDCT_OP.ADD: Evaluator.add,
        NNDCT_OP.SCALAR_ADD: Evaluator.add
        
    }
    # self._redundant_ops = [NNDCT_OP.CONTIGUOUS]

  # def freeze_graph(self):
  #   self._infer_tensor_layout()
  #   self._strip_redundant_ops()
  #   self._constant_folding()
  #   if NndctOption.nndct_parse_debug.value >= 3:
  #     NndctDebugLogger.write(f"\nfrozen dev graph:\n{self._dev_graph}")
   
  def strip_redundant_ops(self):
    # remove unsupported op in xmodel
    redundant_op_types = [NNDCT_OP.CONTIGUOUS]
    self._dev_graph.remove_node_by_types(redundant_op_types)
    
    # remove redundant permute op
    permute_nodes = [node for node in self._dev_graph.nodes if node.op.type == NNDCT_OP.PERMUTE]
    for permute in permute_nodes:
      if permute.node_attr(permute.op.AttrName.ORDER) == [0, 1, 2, 3]:
        self._dev_graph.remove_node(permute)
          
  def constant_folding(self):
    folding_nodes = set()
    for node in self._dev_graph.nodes:
      if node.in_quant_part is False:
          continue
      if hasattr(node.op, "AttrName"):
        for attr_name in node.op.attrs.keys():
          attr_val = node.node_attr(attr_name)
          if isinstance(attr_val, list):
            for i, val in enumerate(attr_val):
              attr_val[i] = self._materialize(node, val, folding_nodes)
          else:
            attr_val = self._materialize(node, attr_val, folding_nodes)
          if node.op.attrs[attr_name].type == list:
            attr_val = [attr_val]
          node.set_node_attr(attr_name, attr_val)
          
    if folding_nodes:   
      for node_name in folding_nodes:
        self._dev_graph.remove_node_forcely(self._dev_graph.node(node_name))
      
      self._dev_graph.reconnect_nodes()

  @staticmethod
  def _infer_op_value_immediately(op_type):
    return op_type in [NNDCT_OP.SHAPE, NNDCT_OP.CONST]

  def _eval_node_value(self, node):
    if node.out_tensors[0].data is None:
      self._evalute_func_map[node.op.type](node)
      
  def _materialize(self, cur_node, value, folding_nodes):
    visited = set()

    def dfs(node):
      visited.add(node.name)
      if self._infer_op_value_immediately(node.op.type):
        folding_nodes.add(node.name)
        self._eval_node_value(node)
        return True
      elif node.name in folding_nodes:
        self._eval_node_value(node)
        return True
      elif node.op.type not in self._evalute_func_map:
        return False

      find_evaluable_op = False
      for tensor in node.in_tensors:
        if tensor.node and tensor.node.name not in visited:  # and tensor.data is None:
          find_evaluable_op = dfs(tensor.node)
          if find_evaluable_op is False:
            break
      
      if find_evaluable_op:
        folding_nodes.add(node.name)
        self._eval_node_value(node)
        
      return find_evaluable_op

    if not isinstance(value, Tensor):
      return value
    else:
      is_evaluable = dfs(value.node)
      if is_evaluable:
        data = value.node.out_tensors[0].data
        cur_node.in_tensors.remove(value)
        if not cur_node.in_tensors and cur_node.op.type not in [NNDCT_OP.ZEROS]:
          folding_nodes.add(cur_node.name)
        return data
      else:
        return value
    
  def infer_tensor_layout(self):
    # TODO: Don't support NHWC in pytorch inference
    for node in self._dev_graph.nodes:
      
      if node.op.type == NNDCT_OP.PERMUTE and node.in_tensors[0].ndim == 4:
        if node.in_tensors[0].layout is None:
          if node.node_attr(node.op.AttrName.ORDER) == [0, 1, 2, 3]:
            node.out_tensors[0].layout = Tensor.Layout.NHWC
        else:
          node.out_tensors[0].layout = node.in_tensors[0].layout
      elif node.out_tensors and node.in_tensors:
        node.out_tensors[0].layout = node.in_tensors[0].layout
      else:
        continue
      
  def partition_by_quant_part(self) -> List[Graph]:
    if not any([node.op.type == NNDCT_OP.QUANT_STUB for node in self._dev_graph.nodes]):
      return [self._dev_graph]
    
    id2nodes = defaultdict(set)
        
    def collect_node_set(node, set_id, visited=None):
      # if not node.in_quant_part:
      #   return
      if not visited:
        visited = []
        
      if not hasattr(node, "set_id"):
        node.set_id = set_id
        
      id2nodes[set_id].add(node)
      visited.append(node)
      
      for cn in self._dev_graph.children(node):
        if cn not in visited and cn.in_quant_part:
          collect_node_set(cn, set_id, visited)  
          
    def get_set_id_from_nodeset(nodeset):
      return min([node.set_id for node in nodeset])
    
    def partition_check(quant_graphs):
      for node in self._dev_graph.nodes:
        if node.in_quant_part and all([node not in graph for graph in quant_graphs]):
          raise RuntimeError(f"Please check graph partition: the quant node '{node.name}' should be in quant graph.")
        elif not node.in_quant_part and any([node in graph for graph in quant_graphs]):
          raise RuntimeError(f"Please check graph partition: the non-quant node '{node.name}' included in quant graph.")   
    
    set_id = 0           
    for node in self._dev_graph.nodes:
      if node.op.type == NNDCT_OP.QUANT_STUB or (not node.in_nodes and node.in_quant_part):
        collect_node_set(node, set_id)
        set_id += 1
    
    merged_id2nodes = defaultdict(set)
    for nodeset in id2nodes.values():
      id = get_set_id_from_nodeset(nodeset)
      merged_id2nodes[id].update(nodeset)
    
    quant_dev_graph = []
    for graph_id, nodes in merged_id2nodes.items():
      subgraph = Graph.create_subgraph_from_nodeset(self._dev_graph, nodes, f"{self._dev_graph.name}_{graph_id}")
      quant_dev_graph.append(subgraph)

    partition_check(quant_dev_graph)
    return quant_dev_graph


  @property
  def frozen_graph(self):
    return self._dev_graph
"""    
  def partition_by_quant_part(self) -> List[Graph]:
    if not any([node.op.type == NNDCT_OP.QUANT_STUB for node in self._dev_graph.nodes]):
      return [self._dev_graph]
  
    id2nodes: Dict[int, List[Node]] = defaultdict(list)
    count = -1   
    start_quant = False
    for node in self._dev_graph.nodes:
      if not node.in_quant_part:
        start_quant = False
        continue
      
      if node.op.type == NNDCT_OP.QUANT_STUB and start_quant is False:
        start_quant = True
        count += 1
        id2nodes[count].append(node)
         
      elif start_quant:
        id2nodes[count].append(node)
    
    quant_dev_graph = []
    for graph_id, nodes in id2nodes.items():
      graph = Graph(f"{self._dev_graph.name}_{graph_id}")
      for node in nodes:
        graph.add_node(node)         
      quant_dev_graph.append(graph)
    
    return quant_dev_graph
"""        
      
  
      
        
      

    
    

        
      
    
