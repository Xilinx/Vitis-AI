# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Mapping, Any, Union, Tuple
import collections

from nndct_shared.base.key_names import NNDCT_OP as OpTypes
from nndct_shared.nndct_graph.base_node import Node
from nndct_shared.metaclass import Singleton
from nndct_shared.nndct_graph.base_graph import Graph
from nndct_shared.pruning import errors
from nndct_shared.pruning import logging
from nndct_shared.pruning import node_group as node_group_lib
from nndct_shared.utils import registry

OPS_WITH_PARAMETERS = [
    OpTypes.CONV2D, OpTypes.CONV3D, OpTypes.CONVTRANSPOSE2D,
    OpTypes.CONVTRANSPOSE3D, OpTypes.DEPTHWISE_CONV2D, OpTypes.DEPTHWISE_CONV3D,
    OpTypes.DEPTHWISE_CONVTRANSPOSE2D, OpTypes.DEPTHWISE_CONVTRANSPOSE3D,
    OpTypes.DENSE, OpTypes.BATCH_NORM, OpTypes.INSTANCE_NORM,
    OpTypes.SEPARABLECONV2D
]

CONV_OPS = [
    OpTypes.CONV2D, OpTypes.CONVTRANSPOSE2D, OpTypes.CONV3D,
    OpTypes.CONVTRANSPOSE3D, OpTypes.SEPARABLECONV2D
]

DEPTHWISECONV_OPS = [
    OpTypes.DEPTHWISE_CONV2D, OpTypes.DEPTHWISE_CONVTRANSPOSE2D,
    OpTypes.DEPTHWISE_CONV3D, OpTypes.DEPTHWISE_CONVTRANSPOSE3D
]
TRANSPOSECONV_OPS = [OpTypes.CONVTRANSPOSE2D, OpTypes.CONVTRANSPOSE3D]

SEPARABLECONV_OPS = [OpTypes.SEPARABLECONV2D]

ALL_CONV_OPS = CONV_OPS + DEPTHWISECONV_OPS + TRANSPOSECONV_OPS

# Operations that cannot receive pruned tensor as input since it is not
# possible to do shape inference for these operations.
_DISALLOW_PRUNED_INPUT_OPS = [
    OpTypes.CHUNK,
    # OpTypes.RESIZE,
    OpTypes.STRIDED_SLICE,
    OpTypes.RESHAPE,
    OpTypes.ZEROS,
]

def is_grouped_conv(op):
  if op.type not in CONV_OPS:
    return False

  group = op.attr['group']
  return group != 1

def is_depthwise_conv(op):
  if op.type in DEPTHWISECONV_OPS:
    return True

  if not is_grouped_conv(op):
    return False

  out_channels = op.attr['out_dim']
  in_channels = op.attr['in_dim']
  group = op.attr['group']
  return group == in_channels

def is_transpose_conv(op):
  if op.type in TRANSPOSECONV_OPS:
    return True
  return False

def is_separable_conv(op):
  if op.type in SEPARABLECONV_OPS:
    # at present
    return True
  return False

def is_separable_conv_depthwise_weight(param):
  if param.name == 'DEPTHWISE_WEIGHT':
    return True
  else:
    return False

def is_separable_conv_pointwise_weight(param):
  if param.name == 'POINTWISE_WEIGHT':
    return True
  else:
    return False

def get_conv_weight(node):
  if node.op.type == OpTypes.SEPARABLECONV2D:
    # pointwise weight can infer the out dim prune
    weight = node.op.param['pointwise_weight']
  else:
    weight = node.op.param['weights']
  return weight

class NodeGroup(object):

  def __init__(self, nodes: List[str], num_groups: int = 1) -> None:
    self._nodes = nodes
    self._num_groups = num_groups

  @property
  def nodes(self) -> List[str]:
    return self._nodes

  @property
  def num_groups(self) -> int:
    return self._num_groups

  @num_groups.setter
  def num_groups(self, v: int):
    assert self._num_groups == 1 or v == self._num_groups, "num_groups can be set to non-one value ONLY ONCE!"
    self._num_groups = v

  def serialize(self):
    return {"nodes": self._nodes, "num_groups": self._num_groups}

  @classmethod
  def deserialize(cls, data: Mapping[str, Any]):
    return cls(data["nodes"], data["num_groups"])

class PrunableGroup(
    collections.namedtuple('PrunableGroup', ['nodes', 'ratio', 'num_groups'])):
  """A group of nodes to be pruned by given ratio."""

  def __eq__(self, other):
    if len(self.nodes) != len(other.nodes):
      return False

    for index in range(len(self.nodes)):
      if self.nodes[index] != other.nodes[index]:
        return False

    return self.ratio == other.ratio

  def serialize(self):
    return {
        'nodes': self.nodes,
        'ratio': self.ratio,
        'num_groups': self.num_groups
    }

  @classmethod
  def deserialize(cls, data: Mapping[str, Any]):
    return cls(data['nodes'], data['ratio'], data.get('num_groups', 1))

class PruningConfig(metaclass=Singleton):

  def __init__(self, channel_divisible: int = 2):
    self._channel_divisible = channel_divisible

  @property
  def channel_divisible(self):
    return self._channel_divisible

  @channel_divisible.setter
  def channel_divisible(self, channel_divisible):
    if channel_divisible <= 0:
      raise errors.OptimizerInvalidArgumentError(
          "'channel_divisible' must be positive.")
    self._channel_divisible = channel_divisible

  def __str__(self):
    return 'PruningConfig(channel_divisible={})'.format(self._channel_divisible)

  def init_from_dict(self, dict_obj: dict) -> None:
    assert isinstance(dict_obj, dict), "dict_obj must be instance of dict"
    if "channel_divisible" in dict_obj:
      self.channel_divisible = int(dict_obj["channel_divisible"])

class PruningSpec(object):
  """Specification indicates how to prune the network."""

  def __init__(self,
               groups=None,
               channel_divisible: int = 2,
               graph_digest: str = None):
    self._node_to_group = {}
    self._groups = []
    self._channel_divisible: int = channel_divisible
    self._graph_digest: str = graph_digest

    if groups:
      for group in groups:
        self.add_group(group)

  def __str__(self):
    return "PruningSpec(groups=%s, channel_divisible=%s)" % (", ".join(
        [str(group) for group in self._groups]), str(self._channel_divisible))

  def __eq__(self, other):
    if len(self.groups) != len(other.groups):
      return False

    for index in range(len(self.groups)):
      if self.groups[index] != other.groups[index]:
        return False
    return True

  @classmethod
  def from_node_groups(cls, groups: List[NodeGroup], ratio: Union[float, Tuple,
                                                                  List]):
    is_tuple_or_list = isinstance(ratio, (tuple, list))
    if is_tuple_or_list:
      assert len(groups) == len(ratio)

    spec = cls()
    for index, group in enumerate(groups):
      if is_tuple_or_list:
        spec.add_group(
            PrunableGroup(group.nodes, ratio[index], group.num_groups))
      else:
        spec.add_group(PrunableGroup(group.nodes, ratio, group.num_groups))
    return spec

  @property
  def channel_divisible(self):
    return self._channel_divisible

  @channel_divisible.setter
  def channel_divisible(self, channel_divisible: int) -> None:
    self._channel_divisible = channel_divisible

  @property
  def graph_digest(self) -> str:
    return self._graph_digest

  @graph_digest.setter
  def graph_digest(self, graph_digest: str) -> None:
    self._graph_digest = graph_digest

  @property
  def groups(self):
    return self._groups

  def add_group(self, group):
    if not isinstance(group, PrunableGroup):
      raise ValueError("Must add a PrunableGroup, but got {}.".format(
          type(group)))

    self._groups.append(group)
    for node_name in group.nodes:
      self._node_to_group[node_name] = group

  def group(self, node_name):
    return self._node_to_group.get(node_name, None)

  def serialize(self):
    data = {
        "groups": [],
        "channel_divisible": self._channel_divisible,
        "graph_digest": self._graph_digest
    }
    for group in self.groups:
      data['groups'].append(group.serialize())
    return data

  @classmethod
  def deserialize(cls, data):
    groups = [PrunableGroup.deserialize(group) for group in data['groups']]
    return cls(groups, data.get("channel_divisible"), data.get("graph_digest"))

class StructuredPruning(object):
  """A data class that saves the structured pruning info of node."""

  UNKNOWN = 0

  def __init__(self,
               node_name,
               ratio=None,
               removed_outputs=None,
               out_dim=UNKNOWN,
               removed_inputs=None,
               in_dim=UNKNOWN,
               master=False):
    self.node_name = node_name
    self.ratio = ratio
    self.removed_outputs = removed_outputs if removed_outputs else []
    self.out_dim = out_dim
    self.removed_inputs = removed_inputs if removed_inputs else []
    self.in_dim = in_dim
    self.master = master
    # for TF separableconv
    # separableconv = [depthwise_conv , pointwise_conv]
    self.removed_separableconv_pointwise_inputs = []

  def __str__(self):
    return "StructuredPruning({}, ratio={}, in_dim={}, out_dim={})".format(
        self.node_name, self.ratio, self.in_dim, self.out_dim)

  def __repr__(self):
    return ("StructuredPruning<name={}, ratio={}, removed_inputs={}, "
            "in_dim={}, removed_outputs={}, out_dim={}>").format(
                self.node_name, self.ratio, self.removed_inputs, self.in_dim,
                self.removed_outputs, self.out_dim)

  def _has_valid_dim(self, name):
    assert name in ['in', 'out']
    dim = getattr(self, name + '_dim')
    return dim is not None and dim != StructuredPruning.UNKNOWN

  def has_out_dim(self):
    return self._has_valid_dim('out')

  def has_in_dim(self):
    return self._has_valid_dim('in')

  def set_unknown_dim(self, name):
    assert name in ['in', 'out']
    setattr(self, name + '_dim', UNKNOWN)

  def serialize(self):
    return {
        'node': self.node_name,
        'ratio': self.ratio,
        'removed_outputs': self.removed_outputs,
        'out_dim': self.out_dim,
        'removed_inputs': self.removed_inputs,
        'in_dim': self.in_dim,
        'master': self.master
    }

  @classmethod
  def deserialize(cls, data):
    return cls(data['node'], data['ratio'], data['removed_outputs'],
               data['out_dim'], data['removed_inputs'], data['in_dim'],
               data['master'])

def group_nodes_by_shared_weight(graph, node_group_union):

  def get_shared_weight_nodes(graph, node_group_union):
    tensor_to_nodes = {}
    for group in node_group_union.groups():
      for node_name in group:
        node = graph.node(node_name)
        for _, tensor in node.op.params.items():
          if tensor.name not in tensor_to_nodes:
            tensor_to_nodes[tensor.name] = []
          tensor_to_nodes[tensor.name].append(node)

    groups = []
    for _, nodes in tensor_to_nodes.items():
      if len(nodes) > 1:
        groups.append(nodes)
    return groups

  groups = get_shared_weight_nodes(graph, node_group_union)
  in_node_groups = []

  for nodes in groups:
    in_nodes = []
    for node in nodes:
      # Find out all prunable ancestor nodes of shared weight nodes
      # and group them together.
      ancestor = find_prunable_ancestor(graph, node)
      if ancestor:
        logging.vlog(
            2,
            'Find prunable ancestor {} from {}:'.format(ancestor.name,
                                                        node.name))
        in_nodes.append(ancestor)
    in_node_groups.append(in_nodes)

  for nodes in groups + in_node_groups:
    for i in range(1, len(nodes)):
      node_group_union.union(nodes[i - 1].name, nodes[i].name)
      logging.vlog(
          2, 'Union ({}, {}) by shared weights'.format(nodes[i - 1].name,
                                                       nodes[i].name))

  return node_group_union

def group_nodes_by_eltwise_op(graph, node_group_union, op_list):
  for node in graph.nodes:
    if node.op.type not in op_list:
      continue

    eltwise_inputs = []
    for name in node.in_nodes:
      input_node = graph.node(name)
      # Depthwise conv must be treated as a slave node.
      if input_node.op.type in CONV_OPS and not is_depthwise_conv(input_node.op):
        eltwise_inputs.append(name)
      else:
        ancestor = find_prunable_ancestor(graph, input_node)
        if ancestor and not is_depthwise_conv(input_node.op):
          eltwise_inputs.append(ancestor.name)
    if len(eltwise_inputs) < 2:
      continue
    logging.vlog(2, "Union ({}, {})".format(eltwise_inputs[0],
                                            eltwise_inputs[1]))
    node_group_union.union(eltwise_inputs[0], eltwise_inputs[1])
  return node_group_union

def get_nodes_cannot_be_pruned(graph, with_group_conv: bool = False):
  nodes_to_exclude = []

  def exclude_node_ancester(node: Node):
    for node_name in node.in_nodes:
      input_node = graph.node(node_name)
      if input_node.op.type in CONV_OPS:
        prunable_node = input_node
      else:
        prunable_node = find_prunable_ancestor(graph, input_node)
      if prunable_node and prunable_node.name not in nodes_to_exclude:
        nodes_to_exclude.append(prunable_node.name)
        logging.vlog(
            3, 'Exclude node "{}" as its output is fed to op "{}"'.format(
                prunable_node.name, node.op.type))

  for node in graph.nodes:
    if is_depthwise_conv(node.op):
      nodes_to_exclude.append(node.name)
    elif is_grouped_conv(node.op) and not with_group_conv:
      nodes_to_exclude.append(node.name)
      logging.vlog(3, 'Exclude grouped conv node "{}"'.format(node.name))
      exclude_node_ancester(node)
    # Exclude input nodes for non-depthwise conv nodes,
    elif node.op.type in _DISALLOW_PRUNED_INPUT_OPS:
      exclude_node_ancester(node)
  return nodes_to_exclude

def assign_num_groups(graph: Graph, groups: List[NodeGroup]) -> None:
  name_group_map: Mapping[str, NodeGroup] = {}
  for group in groups:
    for name in group.nodes:
      name_group_map[name] = group

  for group in groups:
    for name in group.nodes:
      node = graph.node(name)
      if is_grouped_conv(node.op):
        group.num_groups = node.op.attr['group']
        if not is_depthwise_conv(node.op):
          input_node = graph.node(node.in_nodes[0])
          if input_node.op.type in CONV_OPS:
            prunable_node = input_node
          else:
            prunable_node = find_prunable_ancestor(graph, input_node)
          if prunable_node.name in name_group_map:
            name_group_map[
                prunable_node.name].num_groups = node.op.attr['group']

def group_nodes(graph: Graph,
                nodes_to_exclude: List[str] = [],
                with_group_conv: bool = False) -> List[NodeGroup]:
  """Divide convolution nodes into different groups.
  The nodes that connected with each other by elementwise operation
  will be divided into one group.
  """
  node_group_union = node_group_lib.NodeGroupUnion()

  for node in graph.nodes:
    if node.op.type in CONV_OPS:
      node_group_union.add_node(node.name)

  elementwise_op = [OpTypes.ADD, OpTypes.MULTIPLYLAYER, OpTypes.SUB]
  node_group_union = group_nodes_by_eltwise_op(graph, node_group_union,
                                               elementwise_op)
  node_group_union = group_nodes_by_shared_weight(graph, node_group_union)
  nodes_to_exclude.extend(get_nodes_cannot_be_pruned(graph, with_group_conv))

  all_groups: List[List[str]] = node_group_union.groups()
  ret: List[NodeGroup] = []
  for group in all_groups:
    skip = False
    for node in nodes_to_exclude:
      if node in group:
        skip = True
        break
    if not skip:
      ret.append(NodeGroup(group))
  assign_num_groups(graph, ret)
  return ret

def group_nodes_for_ofa_dynamic_conv(graph):
  """Divide convolution nodes into different groups.
  1*1 conv only can expand or squeeze dim
  3*3 conv  0 is_depthwise_conv 0 ancestor node  1 +/* node
            1 not is_depthwise_conv 0 +/* node.
  """
  group_nodes = []

  depthwise_conv_num = 0
  common_conv_num = 0
  is_mobile = -1

  for node in graph.nodes:
    attrs = {name: node.op.get_config(name) for name in node.op.configs}
    if node.op.type in ALL_CONV_OPS and min(attrs['kernel_size'][0],
                                            attrs['kernel_size'][1]) >= 3:
      if is_depthwise_conv(node.op):
        depthwise_conv_num += 1
      else:
        common_conv_num += 1

  if depthwise_conv_num > common_conv_num:  # mobilenet
    is_mobile = 1
  else:
    is_mobile = 0

  for node in graph.nodes:
    attrs = {name: node.op.get_config(name) for name in node.op.configs}
    if is_mobile == 1:  # mobilenet
      if node.op.type in ALL_CONV_OPS and min(attrs['kernel_size'][0],
                                              attrs['kernel_size'][1]) >= 3:
        if is_depthwise_conv(node.op):
          ancestor = find_prunable_ancestor(graph, node)  # 1*1 3*3d 1*1
          if ancestor is not None:
            group_nodes.append([ancestor.name, node.name])
          else:
            child = find_prunable_child(graph, node)  # 3*3d 1*1
            if child is not None:
              group_nodes.append([child.name])
        else:
          group_nodes.append([node.name])
    else:  # resnet
      if node.op.type in ALL_CONV_OPS:
        if is_depthwise_conv(node.op):
          ancestor = find_prunable_ancestor(graph, node)  # 1*1 3*3d 1*1
          if ancestor is not None:
            group_nodes.append([ancestor.name, node.name])
          else:
            child = find_prunable_child(graph, node)  # 3*3d 1*1
            if child is not None:
              group_nodes.append([child.name])
        else:
          group_nodes.append([node.name])

  for node in graph.nodes:
    if node.op.type != OpTypes.ADD and node.op.type != OpTypes.MULTIPLY:
      continue
    eltwise_inputs = []
    for name in node.in_nodes:
      input_node = graph.node(name)
      if input_node.op.type in ALL_CONV_OPS:
        eltwise_inputs.append(name)
      else:
        ancestor = find_prunable_ancestor(graph, input_node, ALL_CONV_OPS)
        if ancestor:
          eltwise_inputs.append(ancestor.name)
    if len(eltwise_inputs
          ) < 2 or eltwise_inputs[0] == eltwise_inputs[1]:  # Exclude hswish
      continue

    for index, group in enumerate(group_nodes):
      if eltwise_inputs[0] in group or eltwise_inputs[1] in group:
        group_nodes[index] = list(set(group + eltwise_inputs))

  # connected components algorithm
  pool = set(map(frozenset, group_nodes))
  groups = []
  while pool:
    groups.append(set(pool.pop()))
    while True:
      for candidate in pool:
        if groups[-1] & candidate:
          groups[-1] |= candidate
          pool.remove(candidate)
          break
      else:
        break

  for index, group in enumerate(groups):
    groups[index] = list(group)

  return groups, is_mobile

def find_leaf_node(graph):
  """
  find first and last conv layer.
  if second_node is depthwise_conv, add it.
  """
  first_conv_nodes = []
  last_conv_nodes = []
  for node in graph.nodes:
    if len(node.in_nodes) == 0:  # input
      first_conv_node = find_prunable_child(graph, node, ALL_CONV_OPS)
      if first_conv_node is not None:
        first_conv_nodes.append(first_conv_node)
        second_conv_node = find_prunable_child(graph, first_conv_node,
                                               ALL_CONV_OPS)
        if second_conv_node is not None and second_conv_node.op.type in DEPTHWISECONV_OPS:
          first_conv_nodes.append(second_conv_node)
    elif len(node.out_nodes) == 0 and node.op.type in ALL_CONV_OPS:
      last_conv_nodes.append(node)
    elif len(node.out_nodes) == 0:  # output
      last_conv_node_list = find_upper_ancestor(graph, node, ALL_CONV_OPS)
      if len(last_conv_node_list) != 0:
        for i in last_conv_node_list:
          last_conv_nodes.append(i)
  return first_conv_nodes, last_conv_nodes

def find_ancestor(graph, node, target_ops, barrier_ops=[]):
  visited = set()
  queue = []

  # start from input nodes
  for input_name in node.in_nodes:
    queue.append(input_name)

  ancestor = None
  while len(queue) != 0:
    node_name = queue.pop(0)
    node = graph.node(node_name)
    if node.op.type in target_ops:
      ancestor = node
      break
    elif node.op.type in barrier_ops:
      continue
    else:
      for input_name in node.in_nodes:
        if input_name not in visited:
          queue.append(input_name)
          visited.add(input_name)
  return ancestor

def find_upper_ancestor(graph, node, target_ops):
  visited = set()

  ancestor_list = []

  def dfs(graph, node, visited, ancestor_list, target_ops):
    if node.op.type in target_ops:
      ancestor_list.append(node)
      return
    visited.add(node.name)
    for input_name in node.in_nodes:
      if input_name not in visited:
        dfs(graph, graph.node(input_name), visited, ancestor_list, target_ops)

  dfs(graph, node, visited, ancestor_list, target_ops)

  return ancestor_list

def find_prunable_ancestor(graph, node, target_ops=CONV_OPS):
  return find_ancestor(graph, node, target_ops, [OpTypes.CONCAT])

def find_child(graph, node, target_ops, barrier_ops=[]):
  visited = set()
  queue = []

  # start from out nodes
  for out_name in node.out_nodes:
    queue.append(out_name)

  child = None
  while len(queue) != 0:
    node_name = queue.pop(0)
    node = graph.node(node_name)
    if node.op.type in target_ops:
      child = node
      break
    elif node.op.type in barrier_ops:
      continue
    else:
      for out_name in node.out_nodes:
        if out_name not in visited:
          queue.append(out_name)
          visited.add(out_name)
  return child

def find_prunable_child(graph, node, target_ops=CONV_OPS):
  return find_child(graph, node, target_ops, [OpTypes.CONCAT])

_op_modifier = registry.Registry("Pruning Modifier Functions")

class RegisterOpModifier(object):
  """A decorator for registering the modification function for an op.
  This decorator can be defined for an op type so that it can infer the op's
  StructuredPruning and modify its attribute by existing pruning result.

  If a modifier for an op is registered multiple times, a KeyError will be
  raised.
  For example, you can define a new modifier for a Conv2D operation
  by placing this in your code:
    @RegisterOpModifier("Conv2D")
    def _modify_conv2d(graph, node, pruning_res):
      ...

  Then in client code you can retrieve the value by making this call:
    pruning_lib.modify_node_by_pruning(graph, node, pruning_res)
  """

  def __init__(self, op_types):
    if not isinstance(op_types, (list, tuple)):
      op_types = [op_types]
    self._op_types = op_types

  def __call__(self, f):
    """Registers f as the statistics function for op_type."""
    if not callable(f):
      raise TypeError("Modification function must be callable.")
    for op_type in self._op_types:
      if not isinstance(op_type, str):
        raise TypeError("op_type must be a string.")
      _op_modifier.register(f, op_type)
    return f

def _set_input_by_upstream(node, pruning_res):
  node_pruning = pruning_res[node.name]
  input_pruning = pruning_res[node.in_nodes[0]]
  node_pruning.removed_inputs = input_pruning.removed_outputs
  node_pruning.in_dim = input_pruning.out_dim
  logging.vlog(
      3,
      'Set input pruning of {} by upstream {}'.format(node.name,
                                                      node.in_nodes[0]))
  return node_pruning

def _set_output_by_input(node, pruning_res):
  node_pruning = pruning_res[node.name]
  node_pruning.removed_outputs = node_pruning.removed_inputs
  node_pruning.out_dim = node_pruning.in_dim
  return node_pruning

def propagate_node_pruning(node, pruning_res):
  _set_input_by_upstream(node, pruning_res)
  _set_output_by_input(node, pruning_res)

def update_node_by_pruning(graph, node, pruning_res):
  """Looks up the node's modification function in the registry and calls it.
  This function takes a NndctGraph object, a NndctNode from it, and the
  dictionary of PruningInfo and if there's an associated modification method,
  calls it. If no function has been registered for the particular op type,
  a general fucntion will be called which simply set current node's pruning info
  and do not update the node's attribute.
  statistics object
  Args:
    graph: A NndctGraph that the pruning is performed on.
    node: A NndctNode describing the operator.
    pruning_res: A dictionary of `StructuredPruning`.
  """

  op_type = node.op.type
  if op_type in _op_modifier:
    mod_func = _op_modifier.lookup(op_type)
    mod_func(graph, node, pruning_res)
  else:
    propagate_node_pruning(node, pruning_res)

def _modify_depthwise(graph, node, pruning_res):
  node_pruning = _set_input_by_upstream(node, pruning_res)
  if not node_pruning.has_in_dim():
    return

  # We do not actively prune the depthwise conv, but simply modify the weights
  # based on the pruning information of the previous layers
  # Weight layout: [depth_multiplier, H, W, input_dim]
  dw_multiplier = node.op.attr['out_dim'] // node.op.attr['in_dim']
  removed_outputs = []
  for c in node_pruning.removed_inputs:
    for k in range(dw_multiplier):
      removed_outputs.append(c * dw_multiplier + k)
  node_pruning.removed_outputs = removed_outputs
  node_pruning.out_dim = node_pruning.in_dim * dw_multiplier

  node.op.attr['group'] = node_pruning.in_dim
  node.op.attr['in_dim'] = node_pruning.in_dim
  node.op.attr['out_dim'] = node_pruning.out_dim

@RegisterOpModifier([
    OpTypes.DEPTHWISE_CONV2D, OpTypes.DEPTHWISE_CONV3D,
    OpTypes.DEPTHWISE_CONVTRANSPOSE2D, OpTypes.DEPTHWISE_CONVTRANSPOSE3D
])
def modify_depthwise_conv(graph, node, pruning_res):
  _modify_depthwise(graph, node, pruning_res)

def modify_separable_conv(graph, node, pruning_res):
  node_pruning = _set_input_by_upstream(node, pruning_res)
  if not node_pruning.has_in_dim():
    return

  dw_multiplier = node.op.attr['depth_multiplier']
  removed_pointwise_kernel_inputs = []
  for c in node_pruning.removed_inputs:
    for k in range(dw_multiplier):
      removed_pointwise_kernel_inputs.append(c * dw_multiplier + k)
  # depthwise kernel can prune inputs but can't prune output
  # pointwise kernel can prune inputs and output
  # but the inputs is depend on the depthwise kernel
  node_pruning.removed_separableconv_pointwise_inputs = \
    removed_pointwise_kernel_inputs
  if node_pruning.has_in_dim():
    node.op.attr['in_dim'] = node_pruning.in_dim

  if node_pruning.has_out_dim():
    node.op.attr['out_dim'] = node_pruning.out_dim

@RegisterOpModifier(CONV_OPS)
def modify_conv2d(graph, node, pruning_res):
  # In pytorch, dw conv is repesented by conv2d with groups == in_channels and
  # out_channels == K * in_channels, where K is a positive integer.
  if is_depthwise_conv(node.op):
    _modify_depthwise(graph, node, pruning_res)
    return

  if is_separable_conv(node.op):
    modify_separable_conv(graph, node, pruning_res)
    return

  node_pruning = _set_input_by_upstream(node, pruning_res)
  if node_pruning.has_in_dim():
    node.op.attr['in_dim'] = node_pruning.in_dim

  if node_pruning.has_out_dim():
    node.op.attr['out_dim'] = node_pruning.out_dim

@RegisterOpModifier([OpTypes.INSTANCE_NORM])
def modify_instancenorm(graph, node, pruning_res):
  # Under test...
  node_pruning = pruning_res[node.name]
  input_pruning = pruning_res[node.in_nodes[0]]

  removed_outputs = input_pruning.removed_outputs
  out_dim = input_pruning.out_dim

  node_pruning.removed_inputs = removed_outputs
  node_pruning.removed_outputs = removed_outputs
  node_pruning.in_dim = out_dim
  node_pruning.out_dim = out_dim

@RegisterOpModifier([OpTypes.BATCH_NORM])
def modify_batchnorm(graph, node, pruning_res):
  node_pruning = pruning_res[node.name]
  input_pruning = pruning_res[node.in_nodes[0]]

  removed_outputs = input_pruning.removed_outputs
  out_dim = input_pruning.out_dim

  node_pruning.removed_inputs = removed_outputs
  node_pruning.removed_outputs = removed_outputs
  node_pruning.in_dim = out_dim
  node_pruning.out_dim = out_dim

  if node_pruning.has_out_dim():
    node.op.attr['out_dim'] = node_pruning.out_dim

@RegisterOpModifier(OpTypes.FLATTEN)
def modify_flatten(graph, node, pruning_res):
  input_pruning = pruning_res[node.in_nodes[0]]
  if input_pruning.removed_outputs:
    downstream_nodes = []
    queue = [node.name]
    while len(queue):
      cur_node = graph.node(queue.pop())
      downstream_nodes.append(cur_node)
      queue.extend(cur_node.out_nodes)

    ancestor = find_prunable_ancestor(graph, node)
    for downstream_node in downstream_nodes:
      if downstream_node.op.type in CONV_OPS:
        raise errors.OptimizerNotExcludeNodeError(
            'Must exclude node from pruning: {}'.format(ancestor.name))

  propagate_node_pruning(node, pruning_res)

@RegisterOpModifier(OpTypes.DENSE)
def modify_dense(graph, node, pruning_res):
  input_pruning = pruning_res[node.in_nodes[0]]
  if not input_pruning.has_out_dim():
    return

  orig_out_depth = len(input_pruning.removed_outputs) + input_pruning.out_dim
  spatial_size = node.op.attr['in_dim'] // orig_out_depth

  # Two data formats:
  # [-1, 7, 7, 64(32)] => [-1, 3136(1518)]
  # [-1, 32(16), 5, 5] => [-1, 800(400)]
  removed_inputs = []
  data_format = graph.data_format if hasattr(
      graph, 'data_format') else 'channels_first'
  if data_format == 'channels_last':
    for s in range(spatial_size):
      for c in input_pruning.removed_outputs:
        removed_inputs.append(s * orig_out_depth + c)
  else:
    for c in input_pruning.removed_outputs:
      for s in range(spatial_size):
        removed_inputs.append(c * spatial_size + s)

  in_features = spatial_size * input_pruning.out_dim
  node.op.attr['in_dim'] = in_features

  node_pruning = pruning_res[node.name]
  node_pruning.removed_inputs = removed_inputs
  node_pruning.in_dim = in_features
  node_pruning.out_dim = node.op.attr['out_dim']

#@RegisterOpModifier(OpTypes.RESHAPE)
#def modify_reshape(graph, node, pruning_res):
#  node_pruning = pruning_res[node.name]
#  input_pruning = pruning_res[node.in_nodes[0]]
#  assert input_pruning.has_out_dim(), "Unknown out_dim of reshape\'s input node"
#
#  shape = node.op.attr['shape']
#  assert len(shape) == 2, "Node {} has an unexpected shape {}".format(
#      node.name, shape)
#  orig_out_depth = len(input_pruning.removed_outputs) + input_pruning.out_dim
#  spatial_size = shape[-1] // orig_out_depth
#
#  # Two data formats:
#  # [-1, 7, 7, 64(32)] => [-1, 3136(1518)]
#  # [-1, 32(16), 5, 5] => [-1, 800(400)]
#  removed_outputs = []
#  input_node = graph.node(node.in_nodes[0])
#  layout = input_node.op.attr['layout']
#  if layout == 'NHWC':
#    for s in range(spatial_size):
#      for c in input_pruning.removed_outputs:
#        removed_outputs.append(s * spatial_size + c)
#  else:
#    for c in input_pruning.removed_outputs:
#      for s in range(spatial_size):
#        removed_outputs.append(c * spatial_size + s)
#
#  node_pruning.removed_outputs = removed_outputs
#  node_pruning.out_dim = spatial_size * input_pruning.out_dim
#
#  pruned_shape = list(shape)
#  pruned_shape[-1] = node_pruning.out_dim
#  node.op.attr['shape'] = pruned_shape

@RegisterOpModifier(OpTypes.CONCAT)
def modify_concat(graph, node, pruning_res):
  out_dim_missing = False
  for tensor in node.in_tensors:
    input_node = tensor.node
    input_pruning = pruning_res[input_node.name]
    if not input_pruning.has_out_dim():
      out_dim_missing = True
      break

  node_pruning = pruning_res[node.name]
  cur_offset = 0
  out_dim = 0
  removed_outputs = []
  for tensor in node.in_tensors:
    input_node = tensor.node
    input_pruning = pruning_res[input_node.name]
    if input_pruning.removed_outputs and out_dim_missing:
      upstream_conv = find_prunable_ancestor(graph, input_node)
      raise errors.OptimizerNotExcludeNodeError(
          'Must exclude node from pruning: {}.'.format(upstream_conv.name))

    if not out_dim_missing:
      for ro in input_pruning.removed_outputs:
        removed_outputs.append(ro + cur_offset)
      out_dim += input_pruning.out_dim
      cur_offset += (len(input_pruning.removed_outputs) + input_pruning.out_dim)
      node_pruning.removed_outputs = removed_outputs
      node_pruning.out_dim = out_dim
  # update removed_inputs & in_dim
  node_pruning.removed_inputs = node_pruning.removed_outputs
  node_pruning.in_dim = node_pruning.out_dim

@RegisterOpModifier(_DISALLOW_PRUNED_INPUT_OPS)
def raise_if_has_pruned_input(graph, node, pruning_res):
  for node_name in node.in_nodes:
    input_pruning = pruning_res[node_name]
    if input_pruning.removed_outputs:
      input_node = graph.node(node_name)
      if input_node.op.type in CONV_OPS:
        prunable_node = input_node
      else:
        prunable_node = find_prunable_ancestor(graph, input_node)
      raise errors.OptimizerNotExcludeNodeError(
          ('Must exclude node from pruning: {}. '
           'Operation "{}" cannot take pruned tensor as input.').format(
               prunable_node.name, node.op.type))
