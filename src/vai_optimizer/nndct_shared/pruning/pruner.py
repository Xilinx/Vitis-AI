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

import abc
import copy
import json
import numpy as np
import os

from typing import List

from nndct_shared.base.key_names import FrameworkType
from nndct_shared.pruning import errors
from nndct_shared.pruning import logging
from nndct_shared.pruning import pruning_lib
from nndct_shared.pruning import utils
from nndct_shared.pruning.pruning_lib import is_depthwise_conv
from nndct_shared.pruning.pruning_lib import is_grouped_conv
from nndct_shared.pruning.utils import generate_indices_group
from nndct_shared.utils import io
from nndct_shared.utils.tensor_util import param_layout_transformer

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

  def _get_channels_to_remove(self,
                              tensor,
                              axis,
                              channel_divisible: int,
                              ratio: float,
                              num_groups: int = 1):
    assert tensor.shape[
        axis] % num_groups == 0, "Number of output_channels must be multiple of num_groups"
    other_axes = tuple(a for a in range(tensor.ndim) if a != axis)
    abs_sum = np.sum(np.absolute(tensor.data), axis=other_axes)
    # Use stable sorting so that pruning result can be compared
    # in regression test.
    removed_channels: List[int] = []
    interval = len(abs_sum) // num_groups
    start_idx = 0
    end_idx = interval
    while start_idx < len(abs_sum):
      sorted_channels = np.argsort(
          abs_sum[start_idx:end_idx], kind='stable').tolist()

      output_depth = len(sorted_channels)
      remain_depth = utils.num_remaining_channels(output_depth, ratio,
                                                  channel_divisible)
      removed_channels += [
          i + start_idx for i in sorted_channels[:output_depth - remain_depth]
      ]
      start_idx = end_idx
      end_idx += interval
    return removed_channels, len(abs_sum) - len(removed_channels)

  def _remove_channels(self,
                       array: np.ndarray,
                       removed_outputs: List[int],
                       removed_inputs: List[int],
                       out_axis: int,
                       in_axis: int,
                       groups: int = 1) -> np.ndarray:
    dim_size = len(array.shape)
    if groups == 1:
      if removed_outputs:
        array = np.delete(array, removed_outputs, axis=out_axis)
      if removed_inputs and dim_size > in_axis:
        array = np.delete(array, removed_inputs, axis=in_axis)
      return array
    else:
      removed_inputs_group = generate_indices_group(removed_inputs, array.shape[in_axis] * groups, groups) \
          if in_axis < dim_size else [[]] * groups
      removed_outputs_group = generate_indices_group(removed_outputs,
                                                     array.shape[out_axis],
                                                     groups)
      parts: List[np.ndarray] = np.split(array, groups, axis=out_axis)
      ret: List[np.ndarray] = []
      for part, removed_i, removed_o in zip(parts, removed_inputs_group,
                                            removed_outputs_group):
        ret.append(
            self._remove_channels(part, removed_o, removed_i, out_axis, in_axis,
                                  1))
      return np.concatenate(ret, axis=out_axis)

  def _generate_pruned_graph(self, node_pruning_results):
    def is_depthwise_conv_from_torch(op):
      assert pruning_lib.is_depthwise_conv(op)
      return not op.has_attr('depth_multiplier')

    graph = self._graph.clone()
    for node in graph.nodes:
      if not node.in_nodes:
        continue
      pruning_lib.update_node_by_pruning(graph, node, node_pruning_results)

    pruned_tensors = set()
    for node in graph.nodes:
      node_pruning = node_pruning_results[node.name]
      for param, tensor in node.op.params.items():
        logging.vlog(
            4,
            'Process {}{} of node {}(op: {})'.format(param, tensor.shape,
                                                     node.name, node.op.type))
        if tensor.name in pruned_tensors:
          # If a module is called for multiple times in forward pass, then
          # the module will be parsed to multiple nodes with same tensor names.
          logging.vlog(
              4, 'Ignore {} of node {} as it has been pruned before.'.format(
                  tensor.name, node.name))
          continue

        org_tensor_shape = tensor.shape
        dim_size = len(tensor.shape)
        if dim_size == 1:
          out_axis, in_axis = 0, dim_size
        elif dim_size == 0:
          continue  # e.g In tf layers.Normalization the parm 'count' is a num
        else:
          out_axis, in_axis = utils.out_in_axis(tensor.ndim)

        ndarray = tensor.data
        removed_outputs = node_pruning.removed_outputs
        removed_inputs = node_pruning.removed_inputs
        # In some rare cases, the (out_channels, in_channles, groups) will be
        # pruned to (16, 1, 1) from (32, 16, 1) and the node will be incorretly
        # considered as depthwise. Use original node attributes to prevent this
        # error.
        original_node = self._graph.node(node.name)
        # Only check depthwise weight (dim_size > 1)
        if pruning_lib.is_depthwise_conv(original_node.op) and dim_size > 1:
          # Depthwise conv weight layout:
          # - nndct: [depth_multiplier, H, W, input_dim]
          # - tensorflow: [H, W, input_dim, depth_multiplier]
          # - torch: [out_channels, 1, H, W], out_channels = depth_multiplier * input_dim

          # Torch parser transpose the original [out_channels, 1, H, W] to
          # [depth_multiplier, H, W, input_dim]
          # (pytorch_nndct/parse/parser.py TorchParser:_load_data)

          # For depthwise original from torch, we use removed_outputs to update weights.
          # For depthwise original from tensorflow, we use removed_inputs to update weights.
          if is_depthwise_conv_from_torch(original_node.op):
            # We tranpose the depthwise weight [depth_multiplier, H, W, input_dim] to
            # [out_channels, H, W, 1], and then we can use removed_outputs to cut out channels.
            # e.g [2, 3, 3, 32] -> [2, 32, 3, 3] -> [64, 1, 3, 3] -> [64, 3, 3, 1]
            ndarray = ndarray.transpose(
                param_layout_transformer(FrameworkType.NNDCT,
                                         FrameworkType.TORCH, tensor.ndim))
            ndarray = ndarray.reshape((-1, 1, *node.op.attr['kernel']))
            ndarray = ndarray.transpose(
                param_layout_transformer(FrameworkType.TORCH,
                                         FrameworkType.NNDCT, tensor.ndim))
            removed_inputs = []
          else:
            removed_outputs = []

        # separateconv  depthwiseweight only can prune inputdim
        # for tf separateconv
        if pruning_lib.is_separable_conv(original_node.op) and \
            pruning_lib.is_separable_conv_depthwise_weight(param):
          removed_outputs = []
        # separateconv  pointwiseweight
        # inputdim prune is determined by depthwiseweight
        if pruning_lib.is_separable_conv(original_node.op) and \
            pruning_lib.is_separable_conv_pointwise_weight(param):
          removed_inputs = node_pruning.removed_separableconv_pointwise_inputs

        logging.vlog(
            4, 'removed_outputs = {}, removed_inputs = {}'.format(
                removed_outputs, removed_inputs))

        ndarray = self._remove_channels(
            ndarray, removed_outputs, removed_inputs, out_axis, in_axis,
            node.op.attr['group'] if is_grouped_conv(node.op) and
            not is_depthwise_conv(node.op) else 1)

        if pruning_lib.is_depthwise_conv(original_node.op) and dim_size > 1 and \
          is_depthwise_conv_from_torch(original_node.op):
          # Recover ndarry to nndct layout: transpose [output_channels, H, W, 1] to
          # [depth_multiplier, H, W, input_dim]
          # Steps: [64, 3, 3, 1] -> [64, 1, 3, 3] -> [2, 32, 3, 3] -> [2, 3, 3, 32]
          ndarray = ndarray.transpose(
              param_layout_transformer(FrameworkType.NNDCT, FrameworkType.TORCH,
                                       tensor.ndim))
          depth_multiplier = int(node.op.attr['out_dim'] /
                                 node.op.attr['in_dim'])
          ndarray = ndarray.reshape(
              (depth_multiplier, -1, *node.node_config('kernel_size')))
          ndarray = ndarray.transpose(
              param_layout_transformer(FrameworkType.TORCH, FrameworkType.NNDCT,
                                       tensor.ndim))

        tensor.from_ndarray(ndarray)
        if tuple(org_tensor_shape) != tuple(tensor.shape):
          pruned_tensors.add(tensor.name)
          logging.vlog(4, 'Reset params done: {}'.format(tensor.shape))

    return graph

  def _validate_node_groups(self, groups):
    """Check if nodes in groups are in graph."""
    for group in groups:
      for node_name in group.nodes:
        if node_name not in self._graph:
          raise errors.OptimizerInvalidAnaResultError(
              ("Node not in graph: {}. "
               "Please confirm you are using correct analysis result."
              ).format(node_name))

  def prune(self, pruning_spec: pruning_lib.PruningSpec):
    self._validate_node_groups(pruning_spec.groups)
    node_pruning_results = {}
    for node in self._graph.nodes:
      pruning_result = pruning_lib.StructuredPruning(node.name)
      # Must set out_dim for all prunable nodes as this is crucial
      # for shape inference.
      if node.op.type in pruning_lib.CONV_OPS:
        pruning_result.out_dim = node.op.attr['out_dim']
      node_pruning_results[node.name] = pruning_result

    for index, group in enumerate(pruning_spec.groups):
      logging.vlog(1, "Group {}: {}".format(index, group))

      master_node = self._graph.node(group.nodes[0])
      weight = pruning_lib.get_conv_weight(master_node)
      out_axis, _ = utils.out_in_axis(weight.ndim)
      removed_outputs, out_dim = self._get_channels_to_remove(
          weight, out_axis, pruning_spec.channel_divisible, group.ratio,
          group.num_groups)

      for name in group.nodes:
        node_pruning_results[name] = pruning_lib.StructuredPruning(
            name, group.ratio, removed_outputs, out_dim)
        logging.vlog(
            3, 'node: {}, removed_outputs: {}, out_dim: {}'.format(
                name, removed_outputs, out_dim))
      node_pruning_results[master_node.name].master = True
      logging.vlog(3,
                   'Group pruning done, master is {}'.format(master_node.name))

    pruned_graph = self._generate_pruned_graph(node_pruning_results)

    return pruned_graph, node_pruning_results

class ModulePruningInfoGenerator(object):

  def __init__(self,
               nodename_to_modulename=None,
               nodes_pruning_info=None,
               module_pruning_info=None):
    self._nodename_to_modulename = nodename_to_modulename
    self._nodes_pruning_info = nodes_pruning_info

    self._module_pruning_info = module_pruning_info
    if not self._module_pruning_info:
      self._module_pruning_info = dict()
      self._generate_module_pruning_info()

  def _generate_module_pruning_info(self):
    for nodename, pruning_info in self._nodes_pruning_info.items():
      if nodename in self._nodename_to_modulename:
        modulename = self._nodename_to_modulename[nodename]
        self._module_pruning_info[modulename] = pruning_info

  @property
  def nodename_to_modulename(self):
    return self._nodename_to_modulename

  @nodename_to_modulename.setter
  def nodename_to_modulename(self, nodename_to_modulename):
    self._nodename_to_modulename = nodename_to_modulename

  @property
  def nodes_pruning_info(self):
    return self._nodes_pruning_info

  @nodes_pruning_info.setter
  def nodes_pruning_info(self, nodes_pruning_info):
    self._nodes_pruning_info = nodes_pruning_info

  @property
  def module_pruning_info(self):
    return self._module_pruning_info

  def serialize(self):
    content = dict()
    for key, value in self._module_pruning_info.items():
      content[key] = value.serialize()

    return content

  @classmethod
  def deserialize(cls, data):
    module_pruning_info = {}
    for key, value in data.items():
      module_pruning_info[key] = pruning_lib.StructuredPruning.deserialize(
          value)
    return cls(module_pruning_info=module_pruning_info)

  def to_json(self):
    return json.dumps(self.serialize(), indent=2)

def save_pruning_info(nodename_to_modulename, nodes_pruning_info, filepath):
  io.create_work_dir(os.path.dirname(filepath))
  with open(filepath, 'w') as f:
    f.write(
        ModulePruningInfoGenerator(nodename_to_modulename,
                                   nodes_pruning_info).to_json())

def load_pruning_info(filepath):
  with open(filepath, 'r') as f:
    content = json.load(f)
  return ModulePruningInfoGenerator.deserialize(content)
