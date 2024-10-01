
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
import copy
import torch
from torch.fx.experimental.optimization import (
    matches_module_pattern,
    replace_node_module)
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_conv_bn_weights
from torch.fx.interpreter import Transformer
from pytorch_nndct.fx.optimization.utils import replace_node

def fetch_attr(target: str, mod):
  target_atoms = target.split(".")
  attr_itr = mod
  for i, atom in enumerate(target_atoms):
      if not hasattr(attr_itr, atom):
          raise RuntimeError(
              f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}"
          )
      attr_itr = getattr(attr_itr, atom)
  return attr_itr


def matches_module_function_pattern(pattern, node, modules):
    if len(node.args) == 0:
        return False
    if not isinstance(node.args[0], torch.fx.Node) or not isinstance(
        node, torch.fx.Node
    ):
        return False
    # the first node is call_module
    if node.args[0].op != "call_module":
        return False
    if not isinstance(node.args[0].target, str):
        return False
    if node.args[0].target not in modules:
        return False
    if type(modules[node.args[0].target]) is not pattern[0]:
        return False
    # the second node is call_function
    if node.op != "call_function":
        return False
    if node.target != pattern[1]:
        return False
    # make sure node.args[0] output is only used by current node.
    if len(node.args[0].users) > 1:
        return False
    return True


def fuse_conv_bn(gm: torch.fx.GraphModule):
  modules_patterns = {
    (torch.nn.Conv1d, torch.nn.BatchNorm1d),
    (torch.nn.Conv2d, torch.nn.BatchNorm2d),
    (torch.nn.Conv3d, torch.nn.BatchNorm3d)
  }
  module_func_patterns = {
    (torch.nn.Conv1d, F.batch_norm),
    (torch.nn.Conv2d, F.batch_norm),
    (torch.nn.Conv3d, F.batch_norm)
  }
  modules = dict(gm.named_modules())
  
  for pattern in modules_patterns:
    for node in gm.graph.nodes:
      if matches_module_pattern(pattern, node, modules):
          if len(node.args[0].users) > 1:  # conv that has multiple consumers is ignored
              continue
          conv = modules[node.args[0].target]
          bn = modules[node.target]
          eval_mode = all(not n.training for n in [conv, bn])
          if not eval_mode:
              continue
          if not bn.track_running_stats:
              continue
          fused_conv = fuse_conv_bn_eval(conv, bn)
          replace_node_module(node.args[0], modules, fused_conv)
          node.replace_all_uses_with(node.args[0])
          gm.graph.erase_node(node)
          gm.graph.lint()
  
  for pattern in module_func_patterns:
    for node in gm.graph.nodes:
      if matches_module_function_pattern(pattern, node, modules):
        # TODO: support kwargs.
        if len(node.args) != 8:
            continue
        conv = modules[node.args[0].target]
        bn_training = node.args[5]
        bn_eps = node.args[7]
        if conv.training or bn_training:
            continue
        if type(bn_eps) is not float:
            continue
        bn_args_is_constant = all(
            n.op == "get_attr" and len(n.users) == 1 for n in node.args[1:5]
        )
        if not bn_args_is_constant:
            continue
        bn_running_mean = fetch_attr(node.args[1].target, gm)
        bn_running_var = fetch_attr(node.args[2].target, gm)
        bn_weight = fetch_attr(node.args[3].target, gm)
        bn_bias = fetch_attr(node.args[4].target, gm)
        if bn_running_mean is None or bn_running_var is None:
            continue
        fused_conv = copy.deepcopy(conv)
        fused_conv.weight, fused_conv.bias = fuse_conv_bn_weights(
            fused_conv.weight,
            fused_conv.bias,
            bn_running_mean,
            bn_running_var,
            bn_eps,
            bn_weight,
            bn_bias,
        )
        replace_node_module(node.args[0], modules, fused_conv)
        node.replace_all_uses_with(node.args[0])
        gm.graph.erase_node(node)
        gm.graph.lint()
  gm.recompile()
  return gm






class QuantizeModule(Transformer):
  def __init__(self, gm, nndct_graph):
    super().__init__(gm)
    self.nndct_graph = nndct_graph

  def run_node(self, n):
      args, kwargs = self.fetch_args_kwargs_from_env(n)
      assert isinstance(args, tuple)
      assert isinstance(kwargs, dict)
      nndct_node = self.nndct_graph.node("::".join([self.nndct_graph.name, n.name]))
      return getattr(self, n.op)(n, nndct_node, args, kwargs)

  def placeholder(self, n, nndct_node, args, kwargs):
    from pytorch_nndct.nn.modules.prim_ops import deephi_Input

    input_proxy = super().placeholder(n.target, args, kwargs)
    new_module = self.create_nndct_node_module(nndct_node, deephi_Input, {})
    return self.tracer.call_module(new_module, new_module.forward, (input_proxy,), {})

  def call_module(self, n, nndct_node, args, kwargs):

    from pytorch_nndct.nn.modules.conv import deephi_Conv2d
    from pytorch_nndct.nn.modules.linear import deephi_Linear

    quant_module_class_map = {
      torch.nn.Conv2d: deephi_Conv2d,
      torch.nn.Linear: deephi_Linear
    }

    mod = self.tracer.root.get_submodule(n.target)
    if mod.__class__ in quant_module_class_map:
      module_initial_kwargs = {config: nndct_node.node_config(config) for config in nndct_node.op.configs}
      new_module = self.create_nndct_node_module(nndct_node, quant_module_class_map[mod.__class__], module_initial_kwargs)
      for name, parameter in new_module.named_parameters():
        parameter.data.copy_(mod.get_parameter(name).data)
        
      for name, buffer in new_module.named_buffers():
        buffer.data.copy_(mod.get_buffer(name).data)

      return self.tracer.call_module(new_module, new_module.forward, args, kwargs)
    else:
      return super().call_module(n.target, args, kwargs)

  def call_function(self, n, nndct_node, args, kwargs):
    from pytorch_nndct.nn.modules.module_template import DeephiCallFunctionModule
    new_module = self.create_nndct_node_module(nndct_node, DeephiCallFunctionModule, {"caller": n.target})
    return self.tracer.call_module(new_module, new_module.forward, args, kwargs)
  
  def call_method(self, n, nndct_node, args, kwargs):
    return super().call_method(n.target, args, kwargs)

  def output(self, n, nndct_node, args, kwargs):
    return super().output(n.target, args, kwargs)

  def create_nndct_node_module(self, nndct_node, module_class, module_initial_kwargs):
    new_module_name = nndct_node.name
    new_module = module_class(**module_initial_kwargs)
    # new_module.node = nndct_node
    new_module.attached_node_name = nndct_node.name
    self.tracer.root.add_submodule(new_module_name, new_module)
    return new_module



  
  

