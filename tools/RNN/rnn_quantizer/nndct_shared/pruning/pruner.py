

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import copy
import numpy as np

from nndct_shared.base import key_names
from nndct_shared.base.key_names import NNDCT_OP
from nndct_shared.pruning import logging
from nndct_shared.pruning import pruning_lib
from nndct_shared.pruning import utils
from nndct_shared.utils import tensor_util

class Pruner(object):
  """Base class for pruners."""

  def __init__(self, graph, *args, **kwargs):
    self._graph = graph

  @abc.abstractmethod
  def prune(self, pruning_spec):
    raise NotImplementedError('Calling an abstract method.')

class ChannelPruner(Pruner):
  """Pruner that performs channel pruning."""

  def __init__(self, graph, *args, **kwargs):
    super(ChannelPruner, self).__init__(graph, *args, **kwargs)

  def _num_remain_channels(self, out_dim, channel_batch, sparsity):
    channel_batch = channel_batch - channel_batch % 2
    channel_batch = int(min(channel_batch, out_dim / 2))
    remain_channels = int((1 - sparsity) * out_dim)
    remain_channels = remain_channels - remain_channels % channel_batch
    return min(max(remain_channels, 2), out_dim)

  def _get_pruned_filters(self, tensor, channel_batch, sparsity):

    def sort_by_norm(tensor):
      filter_axis, _ = utils.tensor_out_in_axis(tensor)
      other_axes = tuple(axis for axis in range(4) if axis != filter_axis)

      filter_sum = np.sum(np.absolute(tensor.data), axis=other_axes)
      # Use mergesort to have a stable sorting.
      # Making this choice doesn't really affect the results of pruning.
      # Mainly for consistency with torch.Tensor.argsort so that pruning
      # result can be tested.
      sorted_channels = np.argsort(filter_sum, kind='mergesort')
      return sorted_channels.tolist()

    sorted_channels = sort_by_norm(tensor)
    output_depth = len(sorted_channels)
    remain_depth = self._num_remain_channels(output_depth, channel_batch,
                                             sparsity)
    return sorted_channels[:output_depth - remain_depth], remain_depth

  def _generate_pruned_graph(self, node_pruning_results):
    graph = copy.deepcopy(self._graph)
    for node in graph.nodes:
      if node.op.type == NNDCT_OP.INPUT:
        continue
      pruning_lib.update_node_by_pruning(graph, node, node_pruning_results)
    return graph

  def prune(self, pruning_spec):
    for index, group in enumerate(pruning_spec.groups):
      logging.vlog(1, "Group {}: {}".format(index, group))

    node_pruning_results = {}
    for node in self._graph.nodes:
      node_pruning_results[node.name] = pruning_lib.NodePruningResult(node.name)
      if node.op.type != NNDCT_OP.CONV2D:
        continue
      group = pruning_spec.group(node.name)
      if not group:
        # Set out_dim even though the node is not going to be pruned.
        node_pruning_results[node.name].out_dim = node.op.attr['out_dim']
        continue

      removed_outputs = []
      removed_outputs, out_dim = self._get_pruned_filters(
          node.op.param['weights'], pruning_spec.channel_batch, group.sparsity)
      logging.vlog(3, 'node: {}, removed outputs: {}, out_dim: {}'.format(
          node.name, removed_outputs, out_dim))
      for name in group.nodes:
        node_pruning_results[name] = pruning_lib.NodePruningResult(
            name, group.sparsity, removed_outputs, out_dim)
      node_pruning_results[node.name].master = True

    pruned_graph = self._generate_pruned_graph(node_pruning_results)
    for node in pruned_graph.nodes:
      node_pruning = node_pruning_results[node.name]
      for param, tensor in node.op.params.items():
        org_tensor_shape = tensor.shape
        dim_size = len(tensor.shape)
        if dim_size == 1:
          out_axis, in_axis = 0, dim_size
        else:
          out_axis, in_axis = utils.tensor_out_in_axis(tensor)

        # The meaning of OI is not the same in nndct and pytorch.
        # In nndct, 'O' means channel multiplier (out_channels // in_channels)
        # and 'I' means in_channels. However, in pytorch, 'O' is out_channels
        # and 'I' is channel multiplier.
        # For example, the weight shape of depthwise conv in pytorch is
        # (32, 1, 3, 3) while in nndct the shape is (1, 3, 3, 32).
        # The weight data format in nndct is OHWI, that is,
        # (channel_multiplier, height, width, in_channels), we have to exchange
        # out_axis and in_axis so that the correct dimension can be removed.
        if pruning_lib.is_depthwise(node.op):
          out_axis, in_axis = in_axis, out_axis

        ndarray = tensor.data
        if node_pruning.removed_outputs:
          ndarray = np.delete(ndarray, node_pruning.removed_outputs, axis=out_axis)
        if node_pruning.removed_inputs and dim_size > in_axis:
          ndarray = np.delete(ndarray, node_pruning.removed_inputs, axis=in_axis)
        tensor.from_ndarray(ndarray)
        if org_tensor_shape != tensor.shape:
          logging.vlog(4, "Reset param of {}({}) {}: {} -> {}".format(node.name,
              node.op.type, param.name, org_tensor_shape, tensor.shape))

    return pruned_graph, node_pruning_results
