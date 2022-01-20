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

import json
from typing import Dict, List, NoReturn, Union, Sequence
from abc import ABC, abstractmethod, abstractproperty
from nndct_shared import utils as nndct_utils
from nndct_shared.base import NNDCT_KEYS, NNDCT_OP
from nndct_shared.nndct_graph.base_node import Node

class GraphBase(ABC):

  @abstractmethod
  def children(self, node):
    """Get successors of a node in graph

    Returns:
    list: list of successors
    """

  @abstractmethod
  def parents(self, node):
    """Get precessors of a node in graph

    Returns:
    list: list of precessors
    """

  @abstractproperty
  def nodes(self):
    """Yield node in graph according to topo order

    Returns:
    generator: yiled a node when tranverse graph
    """

  @abstractproperty
  def op_types(self):
    """Get all op types in graph

    Returns:
    set: set of op types
    """

class Graph(GraphBase):
  """ Graph object of NNDCT, contain list of NndctNodes.
    That will be used for topology or export to XGraph"""

  def __init__(self, graph_name=None):
    super(Graph, self).__init__()
    self._name = graph_name or 'NndctGraph'

    self._nodes_by_name = {}
    self._nodes_by_id = {}
    self._tensors = {}

    self._end_tensors = []
    self._copy_tensor_map = {}

  def __contains__(self, node_or_name: Union[str, Node]) -> bool:
    if isinstance(node_or_name, str):
      return node_or_name in self._nodes_by_name
    else:
      return node_or_name.name in self._nodes_by_name

  def node(self, name):
    """Return node with the specified name"""
    return self._nodes_by_name.get(name, None)

  def get_node_by_idx(self, idx):
    node = self._nodes_by_id.get(idx, None)
    if node is None:
      for block in self.block_subgraphs():
        node = block.get_node_by_idx(idx)
        if node is not None:
          break
    return node

  def get_input_nodes(self):
    input_nodes = []
    for node in self.nodes:
      if len(self.parents(node)) == 0:
        input_nodes.append(node)
    return input_nodes

  def add_node(self, node: Node) -> None:
    if node.idx == -1:
      node.idx = max([node.idx for node in self.nodes]) + 1

    if node.name in self._nodes_by_name or node.idx in self._nodes_by_id:
      raise ValueError("Node with same name {} or id {} has been added.".format(
          node.name, node.idx))
    self._nodes_by_name[node.name] = node
    self._nodes_by_id[node.idx] = node
    for tensor in node.out_tensors:
      self.add_tensor(tensor)

  def remove_node_forcely(self, node):
    del self._nodes_by_name[node.name]
    del self._nodes_by_id[node.idx]

  def remove_node(self, node: Node) -> None:

    if len(node.in_nodes) != len(node.out_tensors):
      raise RuntimeError(
          f"Can't remove node '{node.name}' in which number of inputs is not equal with that of outputs."
      )

    parents = self.parents(node)
    children = self.children(node)

    parent = parents[0] if len(parents) else None

    if parent:
      # In some cases, op's attribute may refer to a tensor.
      # In order to not to update the attribute after deleting the node,
      # we modify parent's output tensor instead of child's input tensor.
      # For example, the topology is like A -> B -> C and there is a node D
      # refering the output tensor of B, which is B:0. Now we want to delete
      # node B and the directly set the output tensor of A to B:0, then there
      # is no need to update D's attribute.
      # if len(parent.out_nodes) > 1:
      #     return 
      tensorId2node = {}
      for i, tensor in enumerate(node.in_tensors):
        if tensor.is_param_tensor():
          continue
        index = tensor.node.out_tensors.index(tensor)
        node.out_tensors[i].name = tensor.name
        old_out = tensor.node.out_tensors[index]
        tensor.node.out_tensors[index] = node.out_tensors[i]
        tensor.node.out_tensors[index].node = tensor.node
        tensorId2node[i] = tensor.node
        for other in self.children(tensor.node):
          if other is node:
            continue
          in_index = other.in_tensors.index(old_out)
          other.in_tensors[in_index] = tensor.node.out_tensors[index]
          
      children = self.children(node)
      if children:
        for cn in children:
          for i, tensor in enumerate(node.out_tensors):
            if tensor in cn.in_tensors:
              pn = tensorId2node[i]
              index = cn.in_nodes.index(node.name)
              cn.in_nodes[index] = pn.name
              pn.add_out_node(cn.name)
              if node.name in pn.out_nodes:
                pn.out_nodes.remove(node.name)
      else:
        for pn in tensorId2node.values():
          pn.out_nodes.remove(node.name)

    else:
      for child in children:
        for i, input_tensor in enumerate(child.in_tensors):
          if input_tensor.is_param_tensor():
            continue
          if input_tensor.node.name == node.name:
            del child.in_tensors[i]
        child.in_nodes.remove(node.name)

    self.remove_node_forcely(node)

  def remove_node_by_types(self, node_types: List[str]) -> Dict[str, str]:
    if any([node_type in self.op_types for node_type in node_types]):
      nodes_to_remove = []
      for node in self.nodes:
        if node.op.type in node_types:
          nodes_to_remove.append(node)

      for node in nodes_to_remove:
        self.remove_node(node)

  def find_nodes_by_types(self, node_types: List[NNDCT_OP]):
    conv_nodes = []
    for node in self.nodes:
      if node.op.type in node_types:
        conv_nodes.append(node)
      else:
        continue

    return conv_nodes

  def reconnect_nodes(self):
    self._nodes_by_id.clear()
    for idx, node in enumerate(self.nodes):
      node.idx = idx
      self._nodes_by_id[idx] = node
      node.clean_connections()
    self.connect_nodes()

  def connect_nodes(self):
    for nodeA in self.nodes:
      for input_tensor in nodeA.in_tensors:
        for nodeB in self.nodes:
          if nodeB is not nodeA and input_tensor in nodeB.out_tensors:
            #nodeB.outputs.add(input_tensor.node.name)
            nodeB.add_out_node(nodeA.name)
            nodeA.add_in_node(input_tensor.node.name)

  def parents(self, node: Union[Node, str]) -> List[Node]:
    if isinstance(node, str):
      node = self.node(node)
    return [self.node(node_name) for node_name in node.in_nodes]

  def children(self, node: Union[Node, str]) -> List[Node]:
    if isinstance(node, str):
      node = self.node(node)
    return [self.node(node_name) for node_name in node.out_nodes]

  def add_tensor(self, tensor):
    self._tensors[tensor.name] = tensor

  def tensor(self, name):
    return self._tensors.get(name, None)

  def param_tensor(self, name):
    for node in self.nodes:
      for tensor_name, tensor in node.op.params.items():
        if tensor.name == name:
          return tensor

  def add_end_tensor(self, tensor):
    self._end_tensors.append(tensor)

  def __repr__(self):
    strs = ["{}:".format(self.__class__.__name__)]
    for n in sorted(self.nodes, key=lambda n: n.idx):
      strs.append("node {}".format(n))
    return "\n".join(strs)

  def __str__(self):
    return json.dumps(self.description(), indent=4, separators=(',', ': '))

  def description(self):
    graph_des = {}
    graph_des['graph_name'] = f"{self.__class__.__name__}"
    graph_des['nodes'] = []
    for n in sorted(self.nodes, key=lambda n: n.idx):
      graph_des['nodes'].append(n.description())

    return graph_des

  def set_node_id(self, index, node):
    node.idx = index
    self._nodes_by_id[index] = node

  def set_copy_tensor(self, tensor, tensor_copy):
    tensor_name = tensor.name
    tensor_copy_name = tensor_copy.name
    self._copy_tensor_map[tensor_name] = tensor_copy_name

  def get_copy_tensor(self, tensor):
    tensor_name = tensor.name
    if tensor_name in self._copy_tensor_map:
      tensor_copy_name = self._copy_tensor_map[tensor_name]
      return self.param_tensor(tensor_copy_name)
    else:
      return self.param_tensor(tensor_name)

  @classmethod
  def create_subgraph_from_nodeset(cls, origin_graph, nodeset, graph_name):
    """
    create a subgraph from nodeset belong to origin graph

    """
    sorted_nodeset = origin_graph.top_sort_nodeset(nodeset)
    subgraph = cls(graph_name)
    for node in sorted_nodeset:
      subgraph.add_node(node)

    return subgraph

  def top_sort_nodeset(self, nodeset: Sequence[Node]) -> List[Node]:
    visited = []
    sorted_nodes = []

    def dfs(node):
      visited.append(node)
      for cn in self.children(node):
        if cn not in visited and cn in nodeset:
          dfs(cn)
      sorted_nodes.append(node)

    for node in nodeset:
      if node not in visited:
        dfs(node)
    return sorted_nodes[::-1]

  def block_subgraphs(self):
    for node in self.nodes:
      if node.blocks:
        for block in node.blocks:
          yield block

  @property
  def id2node_map(self):
    return self._nodes_by_id

  @property
  def name(self):
    return self._name

  @name.setter
  def name(self, name):
    self._name = name

  @property
  def nodes(self):
    for node in self._nodes_by_name.values():
      yield node

  @property
  def tensors(self):
    for tensor in self._tensors.values():
      yield tensor

  @property
  def end_tensors(self):
    return self._end_tensors

  @end_tensors.setter
  def end_tensors(self, tensors):
    self._end_tensors = tensors

  @property
  def copy_tensor_map(self):
    return self._copy_tensor_map

  @copy_tensor_map.setter
  def copy_tensor_map(self, tensor_map):
    self._copy_tensor_map = tensor_map

  @property
  def inputs(self):
    return [node for node in self.nodes if not node.in_nodes]

  @property
  def outputs(self):
    return [node for node in self.nodes if not node.out_nodes]

  @property
  def op_types(self):
    return {node.op.type for node in self.nodes}
