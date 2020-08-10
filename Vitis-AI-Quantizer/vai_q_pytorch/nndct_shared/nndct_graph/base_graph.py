import os
import json
import numpy as np
from typing import Dict, List, Union, NoReturn
from nndct_shared.base import NNDCT_KEYS
from nndct_shared import utils as nndct_utils

from nndct_shared.nndct_graph.base_node import Node
from nndct_shared.nndct_graph.base_tensor import Tensor

class Graph(object):
  """ Graph object of NNDCT, contain list of NndctNodes.
    That will be used for topology or export to XGraph"""

  def __init__(self, graph_name=None, startnode=None, endnode=None):
    super(Graph, self).__init__()
    self._name = graph_name or 'NndctGraph'

    self._nodes_by_name = {}
    self._nodes_by_id = {}
    self._tensors = {}

    self._end_tensors = []

  def node(self, name):
    """Return node with the specified name"""
    return self._nodes_by_name.get(name, None)

  def get_node_by_idx(self, idx):
    return self._nodes_by_id.get(idx, None)

  def add_node(self, node: Node) -> None:
    if node.idx == -1:
      node.idx = len(self._nodes_by_id)

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
    assert len(node.in_tensors) <= 1, "Can't remove a node that has multiple input tensors: {}".format(
        node.name)
    parents = self.parents(node)
    children = self.children(node)

    parent = parents[0] if len(parents) else None

    if parent:
      assert len(parent.out_tensors) == 1 and len(
          node.in_tensors) == 1, "Cannot remove node: {}".format(node.name)

      # In some cases, op's attribute may refer to a tensor.
      # In order to not to update the attribute after deleting the node,
      # we modify parent's output tensor instead of child's input tensor.
      # For example, the topology is like A -> B -> C and there is a node D
      # refering the output tensor of B, which is B:0. Now we want to delete
      # node B and the directly set the output tensor of A to B:0, then there
      # is no need to update D's attribute.
      if len(node.out_tensors) > 0:
        # Make sure the tensor name is the same as the original.
        node.out_tensors[0].name = parent.out_tensors[0].name
        parent.out_tensors[0] = node.out_tensors[0]
        parent.out_tensors[0].node = parent
      for child in children:
        index = child.in_nodes.index(node.name)
        child.in_nodes[index] = parent.name
        parent.add_out_node(child.name)
      parent.out_nodes.remove(node.name)
    else:
      for child in children:
        for i, input_tensor in enumerate(child.in_tensors):
          if input_tensor.node.name == node.name:
            del child.in_tensors[i]
        child.in_nodes.remove(node.name)

    self.remove_node_forcely(node)
    # del self._nodes_by_name[node.name]
    # del self._nodes_by_id[node.idx]

  def remove_node_by_types(self, node_types: List[str]) -> Dict[str, str]:
    if any([node_type in self.op_types for node_type in node_types]):
      nodes_to_remove = []
      for node in self.nodes:
        if node.op.type in node_types:
          nodes_to_remove.append(node)

      for node in nodes_to_remove:
        self.remove_node(node)

  def reconnect_nodes(self):
    for idx, node in enumerate(self.nodes):
      node.idx = idx
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

  def add_end_tensor(self, tensor):
    self._end_tensors.append(tensor)

  def get_end_tensors(self):
    return self._end_tensors

  def __repr__(self):
    strs = ["{}:".format(self.__class__.__name__)]
    for n in sorted(self.nodes, key=lambda n: n.idx):
      strs.append("node {}".format(n))
    return "\n".join(strs)

  def add_node_with_ctx(self, node, p_nodes=None, c_nodes=None):
    r"""add node and connect it with context
            Args:
                node: the Nndct node to be added
                p_nodes:  list of Nndctnode, each of the node is parent of the node, the parent node should exist
                c_nodes:  list of Nndctnode, each of the node is child of the node, the parent node should exist
            Return:
                None
        """
    self.add_node(node.name, NndctNode=node)
    #cut off the connection between p_nodes and c_nodes:
    self.insert_node_with_ctx(node, p_nodes, c_nodes)

  def insert_node_with_ctx(self, node, p_nodes: List[Node],
                           c_nodes: List[Node]) -> NoReturn:
    r"""insert node and connect it with context
            Args:
                node: the Nndct node to be added
                p_nodes:  list of Nndctnode, each of the node is parent of the node, the parent node should exist
                c_nodes:  list of Nndctnode, each of the node is child of the node, the parent node should exist
            Return:
                None
        """
    if p_nodes and c_nodes:
      children_names = [c.name for c in c_nodes]
      parent_names = [p.name for p in p_nodes]
      for p in p_nodes:
        for cur_c in self.children(p.name):
          if cur_c.name in children_names:
            self.remove_edge(p.name, cur_c.name)
            cur_c.remove_inputs([p.name])
      for c in c_nodes:
        for cur_p in self.parents(c.name):
          if cur_p.name in parent_names:
            self.remove_edge(cur_p.name, c.name)
            c.remove_inputs([cur_p.name])
    if p_nodes:
      for p in sorted(p_nodes, key=lambda n: n.idx):
        self.add_edge(p.name, node.name)
        node.add_input(p.name)
    if c_nodes:
      for c in sorted(c_nodes, key=lambda n: n.idx):
        self.add_edge(node.name, c.name)
        c.add_input(node.name)

  def export(self,
             file_prefix='NndctGen_graph',
             graph_format=False,
             with_params=False,
             prefix=''):
    graph_des = self.as_description(prefix)
    with open(file_prefix + NNDCT_KEYS.XMODEL_SUFFIX, 'w') as f:
      if graph_format:
        f.write(nndct_utils.to_jsonstr(graph_des))
      else:
        f.write(json.dumps(graph_des))
    if with_params:
      if os.path.exists(file_prefix + NNDCT_KEYS.XPARAM_SUFFIX):
        os.remove(file_prefix + NNDCT_KEYS.XPARAM_SUFFIX)
      with nndct_utils.HDFShapedStore(file_prefix +
                                      NNDCT_KEYS.XPARAM_SUFFIX) as store:
        for k, v in self._params.items():
          store.save(k, v.shape, v.data)


  def deepcopy(self, graph_cls=None, with_params=False, prefix=''):
    graph_cls = graph_cls or self.__class__
    other_graph = graph_cls.load(des_or_file=self.as_description(prefix=prefix))
    for p, v in self._params.items():
      if with_params:
        other_graph.set_param(p, v)
      else:
        other_graph.set_param(p, {'shape': v.shape, 'dtype': v.dtype})

    return other_graph

  def set_endnode(self, name):
    self._end_node = name

  def set_startnode(self, name):
    self._start_node = name

  

  def set_node_id(self, index, node):
    node.idx = index
    self._nodes_by_id[index] = node

  

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
    return self._tensors

  #@property
  #def compute_nodes(self):
  #  #if the graph contains 1 node, the node should be set as compute node
  #  for n in self.nodes:
  #    if n not in self.inputs and (len(self.parents(n.name)) > 0 or
  #                                 len(list(self.nodes)) == 1):
  #      yield n

  @property
  def inputs(self):
    return [node for node in self.nodes if not node.in_nodes]

  @property
  def outputs(self):
    return [node for node in self.nodes if not node.out_nodes]

  #@property
  #def startnode(self):
  #  return self._start_node

  #@property
  #def endnode(self):
  #  return self._end_node
  @property
  def op_types(self):
    return {node.op.type for node in self.nodes}