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
import sys
from typing import Dict, List, NoReturn, Union, Sequence
from abc import ABC, abstractmethod, abstractproperty
from nndct_shared import utils as nndct_utils
from nndct_shared.base import NNDCT_KEYS, NNDCT_OP
from .base_node import Node
from .base_block import Block
from .base_tensor import Tensor


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
    
  @abstractproperty
  def get_topological_graph_nodes_list(self):
    """Get topological sorting nodes list

    Returns:
    set: list of nodes sort by topological
    """

  def get_graph_depth(self):
    topological_graph_nodes_list = self.get_topological_graph_nodes_list()
    depth = {}
    max_depth = 0
    # init default depth
    for node in topological_graph_nodes_list:
      depth[node.name] = 0
    input_nodes = [node for node in topological_graph_nodes_list if not node.in_nodes]
    #
    for node in topological_graph_nodes_list:
      if node not in input_nodes:
        for pn in self.parents(node):
          if pn.owning_block is not node.owning_block and pn.name not in depth:
              depth[pn.name] = -1
          depth[node.name] = max([depth[node.name], depth[pn.name] + 1])
          max_depth = max(max_depth, depth[node.name])
    return max_depth


class Graph(GraphBase):
  """ Graph object of NNDCT, contain list of NndctNodes.
    That will be used for topology or export to XGraph"""

  def __init__(self, graph_name=None):
    super(Graph, self).__init__()
    self._name = graph_name or 'NndctGraph'

    self._nodes_by_name = {}
    self._nodes_by_id = {}


    self._end_tensors = []
    self._copy_tensor_map = {}

    self._tensors = {}
    self._blocks = []
    self._param_names = []
    self._top_block = None


  def __contains__(self, node_or_name: Union[str, Node]) -> bool:
    if isinstance(node_or_name, str):
      return node_or_name in self._nodes_by_name
    else:
      return node_or_name.name in self._nodes_by_name

  def __deepcopy__(self, memo):
    raise NotImplementedError("Deep copy is prohibited, use `clone_from` instead.")

  def clone(self):
    graph = self.__class__(self.name)
    graph.clone_from(self)
    return graph

  def clone_from(self, src_graph):
    local_map = {}
    converted_nodes = []
    head_node = self.create_node_from(src_graph.head_node, local_map, converted_nodes)
    return_node = self.create_node_from(src_graph.return_node, local_map, converted_nodes)
    top_block = Block(self, None, head_node, return_node)
    self.set_top_block(top_block)

    self._top_block.clone_from(src_graph.block, local_map, converted_nodes)

  def create_node_from(self, src_node, local_map, converted_nodes):
    node = Node(src_node.name, dtype=src_node.dtype, in_quant_part=src_node.in_quant_part)
    node.owning_graph = self
    node.idx = src_node.idx
    node.scope_name = src_node.scope_name
    node.source_range = src_node.source_range
    node.target_device = src_node.target_device
    converted_nodes.append(src_node.name)
    for out in src_node.out_tensors:
      if out.name in local_map:
        node.add_out_tensor(local_map[out.name])
      else:
        tensor = Tensor(name=out.name)
        tensor.clone_from(out)
        local_map[out.name] = tensor
        node.add_out_tensor(tensor)

    for inp in src_node.in_tensors:
      if inp.name in local_map:
        node.add_in_tensor(local_map[inp.name])
      else:
        tensor = Tensor(name=inp.name)
        tensor.clone_from(inp)
        local_map[inp.name] = tensor
        node.add_in_tensor(tensor)
    node.clone_from(src_node, local_map)
    for src_block in src_node.blocks:
      head_node = self.create_node_from(src_block.input_node, local_map, converted_nodes)
      return_node = self.create_node_from(src_block.return_node, local_map, converted_nodes)
      block = Block(self, node, head_node, return_node)
      block.clone_from(src_block, local_map, converted_nodes)
      node.add_block(block)
    return node



  def node(self, name):
    """Return node with the specified name"""
    return self._nodes_by_name.get(name, None)

  def get_node_by_idx(self, idx):
    node = self._nodes_by_id.get(idx, None)
    assert node is not None
    return node

  def get_input_nodes(self):
    input_nodes = []
    for node in self.nodes:
      if (len(self.parents(node)) == 0) and \
        (node.op.type==NNDCT_OP.INPUT or node.op.type==NNDCT_OP.TUPLE_INPUT):
        input_nodes.append(node)
    return input_nodes
  
  def get_input_tensors(self, input_args):
    input_tensors = []
    graph_name = self.name
    input_nodes = self.get_input_nodes()
    for idx in range(len(input_args)):
      #input_node_name = graph_name + "::input_" + str(idx)
      #input_node = self.node(input_node_name)
      input_node = input_nodes[idx]
      input_tensor = input_node.out_tensors[0]
      if input_node.op.type == NNDCT_OP.INPUT:
        input_tensors.append(input_tensor.name)
      elif input_node.op.type == NNDCT_OP.TUPLE_INPUT:
        for index in range(len(input_args[idx])):
          input_tensor_name = input_tensor.name + '.' + str(index)
          input_tensors.append(input_tensor_name)
    return input_tensors
  
  def get_return_tensors(self):
    return_tensors = []
    for tensor in self.return_node.in_tensors:
      return_tensors.append(tensor.name)
    return return_tensors

  def add_node(self, node: Node) -> None:

    if node.name in self._nodes_by_name:
      return
    if node.idx in self._nodes_by_id and node is not self._nodes_by_id[node.idx]:
      raise RuntimeError(f"The id `{node.idx}` of {node.name} has been added into graph")

    if node.idx == -1:
      # if not self._nodes_by_id:
      #   node._idx = 0
      # else:
        # node._idx = max([node.idx for node in self.all_nodes()]) + 1
      node._idx = -sys.maxsize + len(list(self.all_nodes()))
    self._nodes_by_name[node.name] = node
    self._nodes_by_id[node.idx] = node


  def free_node(self, node):
    node.owning_graph = None
    self._nodes_by_name.pop(node.name)
    self._nodes_by_id.pop(node.idx)


  def remove_node(self, node):
    assert node.in_tensors
    assert len(node.out_tensors) == 1
    out_tensor = node.out_tensors[0]
    inp_tensor = node.in_tensors[0]
    out_tensor.replace_uses_with(inp_tensor)
    node.destroy()


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
    for node in self.all_nodes():
      for _, tensor in node.op.params.items():
        if tensor.name == name:
          return tensor

  def add_end_tensor(self, tensor):
    self._end_tensors.append(tensor)

  def __repr__(self):
    return f"Graph(name={self.name})"

  def __str__(self):
    return json.dumps(self.description(), indent=2, separators=(',', ': '))

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


  @classmethod
  def create_subgraph_from_nodeset(cls, origin_graph, nodeset, graph_name):
    """
    create a subgraph from nodeset belong to origin graph

    """
    assert len(nodeset) >= 2
    sorted_nodeset = origin_graph.top_sort_nodeset(nodeset)
    for node in sorted_nodeset:
      node.remove_from_list()

    subgraph = cls(graph_name)
    sorted_nodeset[0].owning_graph = subgraph
    sorted_nodeset[-1].owning_graph = subgraph
    block = Block(subgraph, None, sorted_nodeset[0], sorted_nodeset[-1])
    subgraph.set_top_block(block)
    if len(sorted_nodeset) > 2:
      for node in sorted_nodeset[1:-1]:
        node.owning_graph = subgraph
        subgraph.append_node(node)

    return subgraph

  @staticmethod
  def top_sort_nodeset(nodeset: Sequence[Node]) -> List[Node]:
    sorted_nodeset = sorted(nodeset, key=lambda n: n.topo_position)
    return sorted_nodeset

  def get_topological_graph_nodes_list(self):
 
    nodes_list = [node for node in self.nodes]
    return Graph.top_sort_nodeset(nodes_list)

  @property
  def name(self):
    return self._name

  @name.setter
  def name(self, name):
    self._name = name

  @property
  def nodes(self):
    return self._top_block.nodes

  @property
  def reverse_nodes(self):
    return self._top_block.reverse_nodes

  @property
  def tensors(self):
    for tensor in self._tensors.values():
      yield tensor

  # TODO: Remove
  @property
  def end_tensors(self):
    return [tensor for tensor in self.return_node.in_tensors]

  # @property
  # def copy_tensor_map(self):
  #   return self._copy_tensor_map

  # @copy_tensor_map.setter
  # def copy_tensor_map(self, tensor_map):
  #   self._copy_tensor_map = tensor_map

  @property
  def inputs(self):
    return [node for node in self.all_nodes() if not node.in_nodes]

  @property
  def outputs(self):
    return [node for node in self.all_nodes() if not node.out_nodes]

  @property
  def op_types(self):
    return {node.op.type for node in self.all_nodes()}


  def append_node(self, node):
    self._top_block.append_node(node)


  def add_param_name(self, param_name):
    if param_name not in self._param_names:
      self._param_names.append(param_name)

  def param_names(self):
    return list(self._param_names)

  @property
  def block(self):
    return self._top_block

  def is_tensor_in_graph(self, tensor_name):
    return True if tensor_name in self._tensors else False

  def update_node_idx(self, node, index):
    self._nodes_by_id[index] = node

  def clear_node_id_map(self):
    self._nodes_by_id.clear()

  def remove_tensor(self, tensor):
    self._tensors.pop(tensor.name)
    if tensor.name in self._param_names:
      self._param_names.remove(tensor.name)

  def insert_node_between_nodes(self, new_node, parent_node, child_node):
    assert parent_node.in_node_list() and child_node.in_node_list()
    assert (parent_node.owning_graph == child_node.owning_graph
            and parent_node.owning_block == child_node.owning_block)

    new_node.owning_block = parent_node.owning_block
    new_node.owning_graph = parent_node.owning_graph
    tensor = Tensor(name=new_node.name, node=new_node)
    new_node.add_out_tensor(tensor)

    out_tensor = None
    offset = None
    for out in parent_node.out_tensors:
      for use in out.uses:
        if use.user is child_node:
          out_tensor = out
          offset = use.offset
          break

    #out_tensor.replace_uses_with(new_node.out_tensors[0])
    child_node.replace_input_at(offset, new_node.out_tensors[0])
    new_node.add_in_tensor(out_tensor)
    new_node.insert_after(parent_node)

  def set_top_block(self, block):
    self._top_block = block

  def add_block(self, block):
    self._blocks.append(block)

  def all_blocks(self):
    return self._blocks

  def all_nodes(self):
    for _, node in self._nodes_by_name.items():
      yield node

  @property
  def head_node(self):
    return self._top_block.input_node

  @property
  def return_node(self):
    return self._top_block.return_node

  def clean_tensors_data(self):
    for tensor in self.tensors:
      tensor.clean_data()


