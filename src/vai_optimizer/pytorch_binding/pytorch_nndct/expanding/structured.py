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

from nndct_shared.expanding.expander import ChannelExpander
from nndct_shared.expanding.spec import ExpandableGroup, ExpandingSpec
from nndct_shared.nndct_graph.base_graph import Graph
from nndct_shared.pruning.pruning_lib import group_nodes
from pytorch_nndct import parse
from torch.nn import Module
import torch
from typing import List, Tuple
import copy
from pytorch_nndct.pruning.structured import _WEIGHTED_OPS
from pytorch_nndct.utils import module_util as mod_util
from copy import deepcopy


class ExpandingRunner(object):
  def __init__(self, model: Module, input_signature: torch.Tensor) -> None:
    self._model: Module = model
    parser = parse.TorchParser()
    self._graph: Graph = parser(model._get_name(), deepcopy(model), input_signature)
    self._input_signature: torch.Tensor = input_signature

  def expand_from_spec(self, expanding_spec: ExpandingSpec) -> Module:
    expander = ChannelExpander(self._graph)
    expanded_graph, node_expand_desc = expander.expand(expanding_spec)
    model = deepcopy(self._model)
    for node in expanded_graph.nodes:
      node_expanding = node_expand_desc[node.name]
      if node_expanding.added_in_channel == 0 and node_expanding.added_out_channel == 0:
        continue
      if node.op.type not in _WEIGHTED_OPS:
        if len(list(node.op.params)) != 0:
          raise RuntimeError('Unsupported op with parameters ({})'.format(
              node.name))
        continue
      module_name = mod_util.module_name_from_node(node)
      if not module_name:
        raise ValueError(
            ('Expandable operation "{}" must be instance of '
             '"torch.nn.Module" (Node: {})').format(node.op.type, node.name))

      module = mod_util.get_module(model, module_name)
      expanded_module = mod_util.create_module_by_node(type(module), node)
      expanded_module.to("cpu")
      mod_util.replace_modules(model, module_name, expanded_module)
    return model

  def expand(self, channel_divisible: int=2, nodes_to_exclude: List[str]=[]) -> Tuple[Module, ExpandingSpec]:
    groups: List[List[str]] = [group.nodes for group in group_nodes(self._graph, nodes_to_exclude, with_group_conv=False)]
    expanding_spec = ExpandingSpec()
    for group in groups:
      expanding_spec.add_group(ExpandableGroup(group, channel_divisible))
    return self.expand_from_spec(expanding_spec), expanding_spec
