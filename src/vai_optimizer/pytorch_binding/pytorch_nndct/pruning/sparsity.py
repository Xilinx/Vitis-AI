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

import copy
import torch

from nndct_shared.pruning import logging
from nndct_shared.pruning import errors

from pytorch_nndct.nn.modules.sparse_ops import SparseConv2d, SparseLinear
from pytorch_nndct.pruning import utils
from pytorch_nndct.utils import module_util as mod_util

class SparsePruner(object):
  """Implements Sparse pruning at the module level."""

  def __init__(self, model, inputs):
    """Concrete example:

    ```python
      inputs = torch.randn([1, 3, 224, 224], dtype=torch.float32).cuda()
      model = MyModel()
      sparse_pruner = SparsePruner(model, inputs)
      sparse_model = sparse_pruner.sparse_model(w_sparsity=0.5,a_sparsity=0,block_size=16)
    ```

    Arguments:
      model (Module): Model to be pruned.
      input_specs(tuple or list): The specifications of model inputs.
    """

    if isinstance(model, torch.nn.DataParallel):
      raise errors.OptimizerDataParallelNotAllowedError(
          'DataParallel object is not allowed.')

    self._model = model
    self._inputs = inputs
    self._to_update_dict_list = []  # all module need to replace
    self._block_size = 16

    self._graph = utils.parse_to_graph(model, inputs)

  def sparse_model(self,
                   w_sparsity=0.5,
                   a_sparsity=0,
                   block_size=16,
                   excludes=None):

    assert w_sparsity in [0, 0.5, 0.75]
    assert a_sparsity in [0, 0.5]

    if a_sparsity == 0:
      if w_sparsity not in [0, 0.5, 0.75]:
        raise ValueError(
            ('When a_sparsity is 0, w_sparsity must be in ({})').format(
                [0, 0.5, 0.75]))
    elif a_sparsity == 0.5:
      if w_sparsity != 0.75:
        raise ValueError(('When a_sparsity is 0.5, w_sparsity must be 0.75'))

    self._block_size = block_size

    self.sparse_config = {
        'w_sparsity': w_sparsity,
        'a_sparsity': a_sparsity,
        'block_size': self._block_size
    }

    logging.info('sparse_config:')
    logging.info(self.sparse_config)

    if w_sparsity == a_sparsity == 0:
      return self._model

    logging.info('replace module to sparse module')

    # find all nn.Conv2d \ nn.Linear export excluded module
    model = copy.deepcopy(self._model)

    excluded_nodes = self._get_exclude_nodes(excludes) if excludes else []

    # first_conv_nodes, last_conv_nodes = pruning_lib.find_leaf_node(self._graph)

    excluded_module_list = []
    for excluded_node in list(set(excluded_nodes)):
      excluded_module = mod_util.module_name_from_node(excluded_node)
      excluded_module_list.append(excluded_module)

    for n, m in model.named_modules():
      to_update_dict = {}
      if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        if n in excluded_module_list:
          logging.info('excluded module:')
          logging.info(n)
          logging.info(m)

        elif isinstance(
            m, torch.nn.Conv2d) and (m.in_channels / m.groups % 16 != 0 or
                                     m.out_channels % 8 != 0):  # for group conv
          logging.warn((
              'Skipping module ({}) of {} in_channels, {} out_channels, {} groups for sparsity pruning.'
          ).format(m, m.in_channels, m.out_channels, m.groups))
        elif isinstance(m, torch.nn.Linear) and ((m.out_features % 8) != 0 or
                                                 (m.in_features % 16) != 0):
          logging.warn((
              'Skipping module ({}) of {} in_features, {} out_features for sparsity pruning.'
          ).format(m, m.in_features, m.out_features))
        else:
          to_update_dict[n] = m
          self._to_update_dict_list.append(to_update_dict)

    logging.info('replace module list:')
    for i in (self._to_update_dict_list):
      logging.info(i)

    # replace all nn.Conv2d \ nn.Linear and reload ckpt
    for idx, to_update_dict in enumerate(self._to_update_dict_list):
      for name, sub_module in to_update_dict.items():
        if isinstance(sub_module, torch.nn.Conv2d):
          sparse_modules = SparseConv2d(
              sub_module.in_channels,
              sub_module.out_channels,
              sub_module.kernel_size,
              sub_module.stride,
              sub_module.padding,
              sub_module.dilation,
              sub_module.groups,
              bias=True if sub_module.bias is not None else False,
              **self.sparse_config)
        elif isinstance(sub_module, torch.nn.Linear):
          sparse_modules = SparseLinear(
              sub_module.in_features,
              sub_module.out_features,
              bias=True if sub_module.bias is not None else False,
              **self.sparse_config)

        mod_util.replace_modules(model, name, sparse_modules, copy_ckpt=True)

    return model

  def export_sparse_model(self, model):

    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DataParallel,
                          torch.nn.parallel.DistributedDataParallel)):
      model = model.module

    sparse_model = copy.deepcopy(model)

    for n, m in model.named_modules():
      if isinstance(m, (SparseConv2d, SparseLinear)):
        if isinstance(m, SparseConv2d):
          nn_modules = torch.nn.Conv2d(
              m.conv.in_channels,
              m.conv.out_channels,
              m.conv.kernel_size,
              m.conv.stride,
              m.conv.padding,
              m.conv.dilation,
              m.conv.groups,
              bias=True if m.conv.bias is not None else False)
        elif isinstance(m, SparseLinear):
          nn_modules = torch.nn.Linear(
              m.linear.in_features,
              m.linear.out_features,
              bias=True if m.linear.bias is not None else False)
        mod_util.replace_modules(sparse_model, n, nn_modules, copy_ckpt=True)

    return sparse_model

  def _get_exclude_nodes(self, excludes):
    return utils.excluded_node_names(self._model, self._graph, excludes)
