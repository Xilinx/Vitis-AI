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

import abc
from typing import Mapping, Tuple
from nndct_shared.expanding.spec import BatchNormStructuredExpanding, InstanceNormStructuredExpanding, \
  DataInsert, ExpandingSpec, GenericStructuredExpanding, StructuredExpanding, WeightedNodeStructuredExpanding
from nndct_shared.expanding.expanding_lib import update_node_by_expanding
from nndct_shared.nndct_graph.base_node import Node
from nndct_shared.nndct_graph.base_tensor import Tensor

from nndct_shared.pruning import utils
from nndct_shared.pruning import pruning_lib

from nndct_shared.nndct_graph.base_graph import Graph
import numpy as np
from typing import List
from nndct_shared.base.key_names import NNDCT_OP as OpTypes


class Expander(object):
  """Base class for pruners."""

  def __init__(self, graph: Graph, *args, **kwargs):
    self._graph = graph

  @abc.abstractmethod
  def expand(self, expanding_spec: ExpandingSpec):
    raise NotImplementedError('Calling an abstract method.')


class ChannelExpander(Expander):
  def __init__(self, graph: Graph, *args, **kwargs):
    super().__init__(graph, *args, **kwargs)

  def _generate_structured_expanding(self, node: Node) -> StructuredExpanding:
    if node.op.type in pruning_lib.ALL_CONV_OPS or node.op.type == OpTypes.DENSE:
      return WeightedNodeStructuredExpanding(node.name)
    elif node.op.type == OpTypes.BATCH_NORM:
      return BatchNormStructuredExpanding(node.name)
    elif node.op.type == OpTypes.INSTANCE_NORM:
      return InstanceNormStructuredExpanding(node.name)
    else:
      return GenericStructuredExpanding(node.name)

  def expand(self, expanding_spec: ExpandingSpec) -> Tuple[Graph, Mapping[str, StructuredExpanding]]:
    node_expand_desc: Mapping[str, StructuredExpanding] = {}
    for node in self._graph.nodes:
      node_expand_desc[node.name] = self._generate_structured_expanding(node)

    for group in expanding_spec.groups:
      for node_name in group.nodes:
        node = self._graph.node(node_name)
        expand_desc = node_expand_desc[node.name]
        added_out_channel = 0 if node.op.attr['out_dim'] % group.channel_divisible == 0 \
          else group.channel_divisible - node.op.attr['out_dim'] % group.channel_divisible

        expand_desc.out_dim = node.op.attr['out_dim'] + added_out_channel
        if added_out_channel > 0:
          expand_desc.add_weight_out_insert(DataInsert(node.op.attr['out_dim'], added_out_channel))
          expand_desc.add_bias_insert(DataInsert(node.op.attr['out_dim'], added_out_channel))
    return self._generate_expanded_graph(node_expand_desc), node_expand_desc

  def _generate_expanded_graph(self, node_expand_desc: Mapping[str, StructuredExpanding]) -> Graph:
    graph = self._graph.clone()
    for node in graph.nodes:
      if not node.in_nodes:
        continue
      update_node_by_expanding(graph, node, node_expand_desc)

    expanded_tensors = set()
    for node in graph.nodes:
      node_expanding = node_expand_desc[node.name]
      if isinstance(node_expanding, WeightedNodeStructuredExpanding):
        self._do_weighted_node_insert(node, node_expanding, expanded_tensors)
      elif isinstance(node_expanding, BatchNormStructuredExpanding):
        self._do_batch_norm_insert(node, node_expanding, expanded_tensors)
      elif isinstance(node_expanding, InstanceNormStructuredExpanding):
        self._do_instance_norm_insert(node, node_expanding, expanded_tensors)
    return graph

  def _insert_data(self, raw_data: np.ndarray, inserts: List[DataInsert], axis: int) -> np.ndarray:
    offset = 0
    shape = list(raw_data.shape)
    ret: List[np.ndarray] = []
    for insert in inserts:
      if offset < insert.position:
        ret.append(raw_data.take(indices=range(offset, insert.position), axis=axis))
      if insert.added_num_channels > 0:
        shape[axis] = insert.added_num_channels
        ret.append(
          insert.added_data.data.astype(raw_data.dtype) if insert.added_data is not None else np.zeros(shape, dtype=raw_data.dtype))
      offset = insert.position
    if offset < raw_data.shape[axis]:
      ret.append(raw_data.take(indices=range(offset, raw_data.shape[axis]), axis=axis))
    return np.concatenate(ret, axis=axis)

  def _get_in_out_axis(self, tensor: Tensor, node: Node) -> Tuple[int, int]:
    dim_size = len(tensor.shape)
    if dim_size == 1:
      out_axis, in_axis = 0, dim_size
    else:
      out_axis, in_axis = utils.out_in_axis(tensor.ndim)

    # In some rare cases, the (out_channels, in_channles, groups) will be
    # pruned to (16, 1, 1) from (32, 16, 1) and the node will be incorretly
    # considered as depthwise. Use original node attributes to prevent this
    # error.
    original_node = self._graph.node(node.name)
    if pruning_lib.is_depthwise_conv(original_node.op):
      out_axis, in_axis = in_axis, out_axis
    return in_axis, out_axis, dim_size

  def _do_weighted_node_insert(self, node: Node, expanding_desc: WeightedNodeStructuredExpanding, expanded_tensors=set()):
    for param, tensor in node.op.params.items():
      if tensor.name in expanded_tensors:
        continue

      org_tensor_shape = tensor.shape
      in_axis, out_axis, dim_size = self._get_in_out_axis(tensor, node)
      ndarray = tensor.data
      if expanding_desc.added_out_channel > 0:
        inserts = expanding_desc.weight_out_inserts \
          if param == node.op.ParamName.WEIGHTS else expanding_desc.bias_inserts
        ndarray = self._insert_data(ndarray, inserts, out_axis)
      if expanding_desc.added_in_channel > 0 and dim_size > in_axis and param == node.op.ParamName.WEIGHTS:
        ndarray = self._insert_data(ndarray, expanding_desc.weight_in_inserts, in_axis)
      tensor.from_ndarray(ndarray)
      if tuple(org_tensor_shape) != tuple(tensor.shape):
        expanded_tensors.add(tensor.name)

  def _do_batch_norm_insert(self, node: Node, expanding_desc: BatchNormStructuredExpanding, expanded_tensors=set()):
    for param, tensor in node.op.params.items():
      if tensor.name in expanded_tensors:
        continue

      org_tensor_shape = tensor.shape
      ndarray = tensor.data
      if expanding_desc.added_out_channel > 0:
        if param == node.op.ParamName.MOVING_MEAN:
          ndarray = self._insert_data(ndarray, expanding_desc.moving_mean_inserts, 0)
        elif param == node.op.ParamName.MOVING_VAR:
          ndarray = self._insert_data(ndarray, expanding_desc.moving_var_inserts, 0)
        elif param == node.op.ParamName.GAMMA:
          ndarray = self._insert_data(ndarray, expanding_desc.gamma_inserts, 0)
        elif param == node.op.ParamName.BETA:
          ndarray = self._insert_data(ndarray, expanding_desc.beta_inserts, 0)
      tensor.from_ndarray(ndarray)
      if tuple(org_tensor_shape) != tuple(tensor.shape):
        expanded_tensors.add(tensor.name)

  def _do_instance_norm_insert(self, node: Node, expanding_desc: InstanceNormStructuredExpanding, expanded_tensors=set()):
    for param, tensor in node.op.params.items():
      if tensor.name in expanded_tensors:
        continue

      org_tensor_shape = tensor.shape
      ndarray = tensor.data
      if expanding_desc.added_out_channel > 0:
        if param == node.op.ParamName.GAMMA:
          ndarray = self._insert_data(ndarray, expanding_desc.gamma_inserts, 0)
        elif param == node.op.ParamName.BETA:
          ndarray = self._insert_data(ndarray, expanding_desc.beta_inserts, 0)
      tensor.from_ndarray(ndarray)
      if tuple(org_tensor_shape) != tuple(tensor.shape):
        expanded_tensors.add(tensor.name)
