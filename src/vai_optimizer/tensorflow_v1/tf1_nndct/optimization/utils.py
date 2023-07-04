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

import tensorflow as tf
from typing import List, Mapping
from tf1_nndct.optimization.constant import OpType
from queue import Queue
import numpy as np


class NodeGroupUnion(object):
  def __init__(self) -> None:
    self._parents = []
    self._nodes = []
    self._node_idx_map = {}
  
  def add_node(self, node: str) -> None:
    self._node_idx_map[node] = len(self._nodes)
    self._parents.append(len(self._nodes))
    self._nodes.append(node)

  def find(self, node: str) -> int:
    assert node in self._node_idx_map
    trace = []
    idx = self._node_idx_map[node]
    parent = self._parents[idx]
    while parent != idx:
      trace.append(idx)
      idx = parent
      parent = self._parents[idx]
    for i in trace:
      self._parents[i] = idx
    return idx

  def union(self, node_a: str, node_b: str) -> None:
    node_a_group_id = self.find(node_a)
    node_b_group_id = self.find(node_b)
    if node_a_group_id != node_b_group_id:
      self._parents[node_b_group_id] = node_a_group_id
  
  def get_groups(self) -> List[List[str]]:
    groups = {}
    for n in self._nodes:
      group_id = self.find(n)
      if group_id not in groups:
        groups[group_id] = [n]
      else:
        groups[group_id].append(n)
    return list(groups.values())


def find_nodes_of_specific_types(graph_def: tf.compat.v1.GraphDef, op_types: List[str]) -> List[tf.compat.v1.NodeDef]:
  ret = []
  for node_def in graph_def.node:
    if node_def.op in op_types:
      ret.append(node_def)
  return ret


def find_ancestor_target_nodes(
    node_def_map: Mapping[str, tf.compat.v1.NodeDef], node_name: str, 
    target_op_types: List[str], barrier_op_types: List[str]=[], find_all: bool=False) -> List[tf.compat.v1.NodeDef]:
  assert node_name in node_def_map
  node = node_def_map[node_name]
  q = Queue()
  q.put(node)
  ret = []
  while not q.empty():
    node = q.get()
    if node.op in barrier_op_types:
      continue
    if node.op in target_op_types:
      ret.append(node)
      if not find_all:
        break
    for inpt in node.input:
      inpt = get_input_node_name(inpt)
      assert inpt in node_def_map
      q.put(node_def_map[inpt])
  return ret


def group_eletwise_add(graph_def: tf.compat.v1.GraphDef, node_def_map: Mapping[str, tf.compat.v1.NodeDef], node_group_union: NodeGroupUnion) -> None:
  for elem_add_node in find_nodes_of_specific_types(graph_def, [OpType.Add, OpType.AddV2]):
    input_conv_nodes = []
    for inpt in elem_add_node.input:
      conv_nodes = find_ancestor_target_nodes(node_def_map, inpt, [OpType.Conv2D], [OpType.Concat, OpType.ConcatV2])
      input_conv_nodes.extend(conv_nodes)
    if len(input_conv_nodes) == 2:
      node_group_union.union(input_conv_nodes[0].name, input_conv_nodes[1].name)


def find_weight_nodes(node: tf.compat.v1.NodeDef, node_def_map: Mapping[str, tf.compat.v1.NodeDef]) -> List[tf.compat.v1.NodeDef]:
  weight_nodes = []
  if len(node.input) == 0:
    return []
  for inpt in node.input[1:]:
    weight_nodes.extend(find_ancestor_target_nodes(
      node_def_map, inpt, 
      [OpType.Variable, OpType.VariableV2, OpType.Const, OpType.Placeholder, OpType.VarHandleOp]))
  return weight_nodes


def group_weight_shared(conv_nodes: List[tf.compat.v1.NodeDef], node_def_map: Mapping[str, tf.compat.v1.NodeDef], node_group_union: NodeGroupUnion):
  weight_node_conv_node_map = {}
  for conv_node in conv_nodes:
    weight_node = find_weight_nodes(conv_node, node_def_map)[0]
    if weight_node.name not in weight_node_conv_node_map:
      weight_node_conv_node_map[weight_node.name] = [conv_node.name]
    else:
      weight_node_conv_node_map[weight_node.name].append(conv_node.name)
  for conv_node_names in weight_node_conv_node_map.values():
    for i in range(1, len(conv_node_names)):
      node_group_union.union(conv_node_names[0], conv_node_names[i])
    
    parent_conv_nodes = []
    for conv_node_name in conv_node_names:
      parent_conv_nodes.extend(find_ancestor_target_nodes(node_def_map, conv_node_name, [OpType.Conv2D], [OpType.Concat, OpType.ConcatV2]))
    for i in range(1, len(parent_conv_nodes)):
      node_group_union.union(parent_conv_nodes[0], parent_conv_nodes[i])


def group_conv_nodes(graph_def: tf.compat.v1.GraphDef, excludes: List[str]) -> List[List[str]]:
  node_def_map = {node_def.name: node_def for node_def in graph_def.node}
  node_group_union = NodeGroupUnion()
  conv_nodes = find_nodes_of_specific_types(graph_def, [OpType.Conv2D])
  for conv_node in conv_nodes:
    node_group_union.add_node(conv_node.name)
  group_eletwise_add(graph_def, node_def_map, node_group_union)
  group_weight_shared(conv_nodes, node_def_map, node_group_union)

  exclude_nodes = set(excludes)
  for invalid_node in find_nodes_of_specific_types(graph_def, [OpType.Reshape]):
    if invalid_node.name.endswith("flatten/Reshape"):
      continue
    for invalid_conv_node in find_ancestor_target_nodes(node_def_map, invalid_node.name, [OpType.Conv2D]):
      exclude_nodes.add(invalid_conv_node.name)
  
  groups = []
  for g in node_group_union.get_groups():
    exclude_current_group = False
    for n in g:
      if n in exclude_nodes:
        exclude_current_group = True
        break
    if not exclude_current_group:
      groups.append(g)
  return groups

def calculate_flops(frozen_graph_def: tf.compat.v1.GraphDef, input_specs: Mapping[str, tf.TensorSpec]) -> float:
  calculation_types = [OpType.Conv2D, OpType.QuantizedConv2D, OpType.MatMul, OpType.QuantizedMatMul, OpType.DepthwiseConv2dNative]
  calculation_nodes = []
  for node_def in frozen_graph_def.node:
    if node_def.op in calculation_types:
      calculation_nodes.append(node_def)
  with tf.compat.v1.Graph().as_default() as g:
    with tf.Session().as_default() as sess:
      tf.graph_util.import_graph_def(frozen_graph_def, name="")
      node_def_map = {n.name: n for n in frozen_graph_def.node}
      feed_dict = {k: sess.run(tf.zeros(v.shape, v.dtype)) for k, v in input_specs.items()}
      calculation_node_tensors = sess.run(
          [g.get_tensor_by_name(n.name + ':0') for n in calculation_nodes], feed_dict=feed_dict)
  
  flops = 0
  for idx, node in enumerate(calculation_nodes):
    if node.op in [OpType.Conv2D, OpType.QuantizedConv2D]:
      weight_node = find_weight_nodes(node, node_def_map)[0]
      filter_height = weight_node.attr['value'].tensor.tensor_shape.dim[0].size
      filter_width = weight_node.attr['value'].tensor.tensor_shape.dim[1].size
      filter_in_depth = weight_node.attr['value'].tensor.tensor_shape.dim[2].size
      flops += calculation_node_tensors[idx].size * filter_height * filter_width * filter_in_depth * 2
    elif node.op in [OpType.MatMul, OpType.QuantizedMatMul]:
      weight_node = find_weight_nodes(node, node_def_map)[0]
      if (node.attr["transpose_a"]):
          k = weight_node.attr['value'].tensor.tensor_shape.dim[0].size
      else:
          k = weight_node.attr['value'].tensor.tensor_shape.dim[1].size
      flops += k * calculation_node_tensors[idx].size * 2
    else:
      weight_node = find_weight_nodes(node, node_def_map)[0]
      filter_height = weight_node.attr['value'].tensor.tensor_shape.dim[0].size
      filter_width = weight_node.attr['value'].tensor.tensor_shape.dim[1].size
      flops += calculation_node_tensors[idx].size * filter_height * filter_width * 2
  return flops


def is_matmul(node_def: tf.compat.v1.GraphDef) -> bool:
  return node_def.op in [OpType.MatMul, OpType.QuantizedMatMul, OpType.BatchMatMul, OpType.SparseMatMul]


def is_conv(node_def: tf.compat.v1.GraphDef) -> bool:
  return node_def.op in [OpType.Conv2D, OpType.QuantizedConv2D]


def is_depthwise_conv(node_def: tf.compat.v1.GraphDef) -> bool:
  return node_def.op == OpType.DepthwiseConv2dNative


def is_concat(node_def: tf.compat.v1.GraphDef) -> bool:
  return node_def.op in [OpType.Concat, OpType.ConcatV2]


def is_biasadd(node_def: tf.compat.v1.GraphDef) -> bool:
  return node_def.op == OpType.BiasAdd


def is_weighted_node(node_def: tf.compat.v1.GraphDef) -> bool:
  return is_conv(node_def) or is_matmul(node_def) or is_depthwise_conv(node_def) or \
      is_biasadd(node_def) or node_def.op == OpType.FusedBatchNormV3


def get_input_node_name(input_name: str) -> str:
  if input_name.startswith("^"):
    input_name = input_name[1:]
  return input_name.split(":")[0]


def topo_sort(graph_def: tf.compat.v1.GraphDef) -> List[tf.compat.v1.NodeDef]:
  reverse_map = {}
  num_inputs = {}
  ret = []
  for node_def in graph_def.node:
    if len(node_def.input) == 0:
      ret.append(node_def)
    else:
      num_inputs[node_def.name] = len(node_def.input)
    for inpt in node_def.input:
      inpt = get_input_node_name(inpt)
      if inpt in reverse_map:
        reverse_map[inpt].append(node_def)
      else:
        reverse_map[inpt] = [node_def]
  
  idx = 0
  while idx < len(ret):
    if ret[idx].name in reverse_map:
      for node in reverse_map[ret[idx].name]:
        num_inputs[node.name] -= 1
        if num_inputs[node.name] == 0:
          ret.append(node)
    idx += 1
  return ret
