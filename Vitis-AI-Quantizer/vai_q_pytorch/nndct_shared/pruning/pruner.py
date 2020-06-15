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
      sorted_channels = np.argsort(filter_sum)
      return sorted_channels.tolist()

    sorted_channels = sort_by_norm(tensor)
    output_depth = len(sorted_channels)
    remain_depth = self._num_remain_channels(output_depth, channel_batch,
                                             sparsity)
    return sorted_channels[:output_depth - remain_depth], remain_depth

  def _generate_pruned_graph(self, pruning_info):
    graph = copy.deepcopy(self._graph)
    for node in graph.nodes:
      if node.op.type == NNDCT_OP.INPUT:
        continue
      pruning_lib.update_node_by_pruning(graph, node, pruning_info)
    return graph

  def prune(self, pruning_spec):
    for index, group in enumerate(pruning_spec.groups):
      logging.vlog(1, "Group {}: {}".format(index, group))

    pruning_info = {}
    for node in self._graph.nodes:
      pruning_info[node.name] = pruning_lib.NodePruningInfo(node.name)
      if node.op.type != NNDCT_OP.CONV2D:
        continue
      group = pruning_spec.group(node.name)
      if not group:
        continue

      removed_outputs = []
      removed_outputs, out_dim = self._get_pruned_filters(
          node.op.param['weight'], pruning_spec.channel_batch, group.sparsity)
      logging.vlog(3, 'Removed output channels of {}: {}'.format(
          node.name, removed_outputs))
      pruning_info[node.name].master = True
      for name in group.nodes:
        pruning_info[name] = pruning_lib.NodePruningInfo(
            name, removed_outputs, out_dim)

    pruned_graph = self._generate_pruned_graph(pruning_info)
    for node in pruned_graph.nodes:
      node_pruning = pruning_info[node.name]
      for param, tensor in node.op.params.items():
        org_tensor_shape = tensor.shape
        dim_size = len(tensor.shape)
        if dim_size == 1:
          out_axis, in_axis = 0, dim_size
        else:
          out_axis, in_axis = utils.tensor_out_in_axis(tensor)
        ndarray = tensor.data
        if node_pruning.removed_outputs:
          ndarray = np.delete(ndarray, node_pruning.removed_outputs, axis=out_axis)
        if node_pruning.removed_inputs and dim_size > in_axis:
          ndarray = np.delete(ndarray, node_pruning.removed_inputs, axis=in_axis)
        tensor.from_ndarray(ndarray)
        if org_tensor_shape != tensor.shape:
          logging.vlog(1, "Reset param of {}({}) {}: {} -> {}".format(node.name,
              node.op.type, param.name, org_tensor_shape, tensor.shape))

    return pruned_graph, pruning_info
