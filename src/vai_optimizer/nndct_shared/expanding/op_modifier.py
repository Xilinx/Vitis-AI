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

from typing import Mapping
from nndct_shared.expanding.spec import BatchNormStructuredExpanding, InstanceNormStructuredExpanding, \
  DataInsert, GenericStructuredExpanding, StructuredExpanding, WeightedNodeStructuredExpanding
from nndct_shared.nndct_graph.base_graph import Graph
from nndct_shared.nndct_graph.base_node import Node
from nndct_shared.nndct_graph.base_tensor import Tensor
from nndct_shared.utils import registry
from nndct_shared.base.key_names import NNDCT_OP as OpTypes
from nndct_shared.pruning.pruning_lib import is_depthwise_conv, _DISALLOW_PRUNED_INPUT_OPS, find_prunable_ancestor, CONV_OPS
import numpy as np


op_modifier = registry.Registry("expanding Modifier Functions")

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
      op_modifier.register(f, op_type)
    return f


def _set_input_by_upstream(node: Node, expanding_desc: Mapping[str, StructuredExpanding]) -> StructuredExpanding:
  node_expanding = expanding_desc[node.name]
  assert isinstance(node_expanding, WeightedNodeStructuredExpanding), \
    "Variable node_expanding here has to be instance of WeightedNodeStructuredExpanding"
  input_expanding = expanding_desc[node.in_nodes[0]]
  node_expanding.in_dim = input_expanding.out_dim
  for weight_insert in input_expanding.out_inserts:
    node_expanding.add_weight_in_insert(
      DataInsert(weight_insert.position, weight_insert.added_num_channels))
  return node_expanding


def _modify_depthwise(graph: Graph, node: Node, expanding_desc: Mapping[str, StructuredExpanding]):
  node_expanding = expanding_desc[node.name]
  assert isinstance(node_expanding, WeightedNodeStructuredExpanding), \
    "Variable node_expanding here has to be instance of WeightedNodeStructuredExpanding"
  input_expanding = expanding_desc[node.in_nodes[0]]

  dw_multiplier = node.op.attr['out_dim'] // node.op.attr['in_dim']

  node.op.attr["group"] += input_expanding.added_out_channel
  node.op.attr['in_dim'] += input_expanding.added_out_channel
  node.op.attr['out_dim'] += input_expanding.added_out_channel * dw_multiplier

  node_expanding.in_dim = node.op.attr['in_dim']
  node_expanding.out_dim = node.op.attr['out_dim']

  for input_insert in input_expanding.out_inserts:
    node_expanding.add_weight_out_insert(
      DataInsert(input_insert.position * dw_multiplier, input_insert.added_num_channels * dw_multiplier))
    node_expanding.add_bias_insert(
      DataInsert(input_insert.position * dw_multiplier, input_insert.added_num_channels * dw_multiplier))


@RegisterOpModifier([
    OpTypes.DEPTHWISE_CONV2D, OpTypes.DEPTHWISE_CONV3D,
    OpTypes.DEPTHWISE_CONVTRANSPOSE2D, OpTypes.DEPTHWISE_CONVTRANSPOSE3D
])
def modify_depthwise_conv(graph, node, pruning_res):
  _modify_depthwise(graph, node, pruning_res)


@RegisterOpModifier([
    OpTypes.CONV2D, OpTypes.CONV3D, OpTypes.CONVTRANSPOSE2D,
    OpTypes.CONVTRANSPOSE3D
])
def modify_conv2d(graph: Graph, node: Node, expanding_desc: Mapping[str, StructuredExpanding]):
  # In pytorch, dw conv is repesented by conv2d with groups == in_channels and
  # out_channels == K * in_channels, where K is a positive integer.
  if is_depthwise_conv(node.op):
    _modify_depthwise(graph, node, expanding_desc)
    return

  assert node.op.attr['group'] == 1, 'Grouped convolution is not allowed.'
  node_expanding = _set_input_by_upstream(node, expanding_desc)
  node.op.attr["in_dim"] += node_expanding.added_in_channel
  node.op.attr["out_dim"] += node_expanding.added_out_channel

  node_expanding.in_dim = node.op.attr["in_dim"]
  node_expanding.out_dim = node.op.attr["out_dim"]


@RegisterOpModifier([OpTypes.INSTANCE_NORM])
def modify_instancenorm(graph: Graph, node: Node, expanding_desc: Mapping[str, StructuredExpanding]):
  # Under test...
  node_expanding = expanding_desc[node.name]
  assert isinstance(node_expanding, InstanceNormStructuredExpanding), \
    "Variable node_expanding here has to be instance of InstanceNormStructuredExpanding"
  input_expanding = expanding_desc[node.in_nodes[0]]

  node_expanding.in_dim = input_expanding.out_dim
  node_expanding.out_dim = input_expanding.out_dim
  node.op.attr["num_features"] = input_expanding.out_dim

  for insert in input_expanding.out_inserts:
    node_expanding.add_gamma_insert(DataInsert(insert.position, insert.added_num_channels))
    node_expanding.add_beta_insert(DataInsert(insert.position, insert.added_num_channels))


@RegisterOpModifier([OpTypes.BATCH_NORM])
def modify_batchnorm(graph: Graph, node: Node, expanding_desc: Mapping[str, StructuredExpanding]):
  node_expanding = expanding_desc[node.name]
  assert isinstance(node_expanding, BatchNormStructuredExpanding), \
    "Variable node_expanding here has to be instance of BatchNormStructuredExpanding"
  input_expanding = expanding_desc[node.in_nodes[0]]

  node_expanding.in_dim = input_expanding.out_dim
  node_expanding.out_dim = input_expanding.out_dim
  node.op.attr["out_dim"] = input_expanding.out_dim

  for insert in input_expanding.out_inserts:
    node_expanding.add_moving_mean_insert(
      DataInsert(insert.position, insert.added_num_channels, Tensor(data=np.zeros(insert.added_num_channels))))
    node_expanding.add_moving_var_insert(
      DataInsert(insert.position, insert.added_num_channels, Tensor(data=np.ones(insert.added_num_channels))))
    node_expanding.add_beta_insert(
      DataInsert(insert.position, insert.added_num_channels, Tensor(data=np.zeros(insert.added_num_channels))))
    node_expanding.add_gamma_insert(
      DataInsert(insert.position, insert.added_num_channels, Tensor(data=np.ones(insert.added_num_channels))))


@RegisterOpModifier(OpTypes.DENSE)
def modify_dense(graph: Graph, node: Node, expanding_desc: Mapping[str, StructuredExpanding]):
  node_expanding = expanding_desc[node.name]
  assert isinstance(node_expanding, WeightedNodeStructuredExpanding), \
    "Variable node_expanding here has to be instance of WeightedNodeStructuredExpanding"
  input_expanding = expanding_desc[node.in_nodes[0]]

  original_input_channels = input_expanding.out_dim - input_expanding.added_out_channel
  spatial_size = node.op.attr["in_dim"] // original_input_channels

  data_format = graph.data_format if hasattr(
      graph, 'data_format') else 'channels_first'
  if data_format == 'channels_last':
    for weight_insert in input_expanding.out_inserts:
      for i in range(spatial_size):
        node_expanding.add_weight_in_insert(
          DataInsert(weight_insert.position + i * original_input_channels, weight_insert.added_num_channels))
  else:
    for weight_insert in input_expanding.out_inserts:
      node_expanding.add_weight_in_insert(
        DataInsert(weight_insert.position * spatial_size, weight_insert.added_num_channels * spatial_size))

  node.op.attr["in_dim"] += node_expanding.added_in_channel


@RegisterOpModifier(OpTypes.CONCAT)
def modify_concat(graph: Graph, node: Node, expanding_desc: Mapping[str, StructuredExpanding]):
  offset = 0
  out_dim = 0
  node_expanding = expanding_desc[node.name]
  assert isinstance(node_expanding, GenericStructuredExpanding), \
    "Variable node_expanding here has to be instance of GenericStructuredExpanding"
  for node in node.in_nodes:
    input_expanding = expanding_desc[node]
    out_dim += input_expanding.out_dim
    for weight_insert in input_expanding.out_inserts:
      node_expanding.add_insert(
        DataInsert(offset + weight_insert.position, weight_insert.added_num_channels))
    offset += input_expanding.out_dim - input_expanding.added_out_channel

  node_expanding.out_dim = out_dim


@RegisterOpModifier(_DISALLOW_PRUNED_INPUT_OPS)
def raise_if_has_pruned_input(graph: Graph, node: Node, expanding_desc: Mapping[str, StructuredExpanding]):
  for node_name in node.in_nodes:
    input_pruning = expanding_desc[node_name]
    if input_pruning.added_out_channel > 0:
      input_node = graph.node(node_name)
      if input_node.op.type in CONV_OPS:
        prunable_node = input_node
      else:
        prunable_node = find_prunable_ancestor(graph, input_node)
      raise RuntimeError(('Operation "{}" cannot take expanded tensor as input, '
                          'please exclude node "{}" from pruning.').format(
                              node.op.type, prunable_node.name))
