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
import torch
import types

from pytorch_nndct.nn.qat.modules.quantizer import FakeQuantizer
from pytorch_nndct.parse.torch_graph import _NODE_NAME_SEPERATOR

def module_name_from_node(node):
  """Extract module name from parsed nndct node.

  Examples:
    A node's name is
    'ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv1]/input.50',
    then extracted module name is 'layer4.1.conv1'
  """
  node_name = node if isinstance(node, str) else node.name
  if _NODE_NAME_SEPERATOR not in node_name:
    return None

  name_parts = []
  parts = node_name.split(_NODE_NAME_SEPERATOR)[1:-1]
  for part in parts:
    if '[' not in part or ']' not in part:
      raise RuntimeError('Can not extract module name from node {}'.format(
          node_name))
    left_bracket = part.index('[')
    right_bracket = part.index(']')
    name_parts.append(part[left_bracket + 1 : right_bracket])
  return '.'.join(name_parts)

def get_module_by_node(model, node):
  module_name = module_name_from_node(node)
  if not module_name:
    return None
  return get_module(model, module_name)

# Generalization of getattr
def get_module(model, submodule_key):
  tokens = submodule_key.split('.')
  cur_mod = model
  for s in tokens:
    cur_mod = getattr(cur_mod, s)
  return cur_mod

# Generalization of setattr
def set_module(model, submodule_key, module):
  tokens = submodule_key.split('.')
  sub_tokens = tokens[:-1]
  cur_mod = model
  for s in sub_tokens:
    cur_mod = getattr(cur_mod, s)

  setattr(cur_mod, tokens[-1], module)

def replace_modules(model, module_names, new_modules):
  if not isinstance(module_names, (tuple, list)):
    module_names = [module_names]
  if not isinstance(new_modules, (tuple, list)):
    new_modules = [new_modules]
  assert len(new_modules) <= len(module_names)

  modules = []
  for name in module_names:
    modules.append(get_module(model, name))

  for handle_id, pre_hook_fn in modules[0]._forward_pre_hooks.items():
    new_modules[0].register_forward_pre_hook(pre_hook_fn)
    del modules[0]._forward_pre_hooks[handle_id]
  # Move post forward hooks of the last module to resulting fused module
  for handle_id, hook_fn in modules[-1]._forward_hooks.items():
    new_modules[-1].register_forward_hook(hook_fn)
    del modules[-1]._forward_hooks[handle_id]

  # Use Identity to fill in the missing modules.
  for i in range(len(module_names) - len(new_modules)):
    new_modules.append(torch.nn.Identity())
    new_modules[i].training = modules[0].training

  for i, name in enumerate(module_names):
    set_module(model, name, new_modules[i])

def setattr_if_has(module, name, attr):
  if not hasattr(module, name):
    raise AttributeError('Attempting to set an non-existing attribute "{}"'.format(name))
  setattr(module, name, attr)

def enable_dump_blob(model, graph=False, mode='print'):

  node_attr = '_dbg_node'
  prefix_attr = '_dbg_prefix'
  inp_attr = '_dbg_input'
  out_attr = '_dbg_output'
  saver_attr = '_saver_hooked'

  def save_io(module, input, output):
    setattr(module, inp_attr, input)
    setattr(module, out_attr, output)

  def print_saved_io(module):
    saved_inputs = getattr(module, inp_attr, None)
    saved_outputs = getattr(module, out_attr, None)
    if saved_inputs is None or saved_outputs is None:
      print('[WARN] No saved blob: node={}, module={}\n'.format(node_name, module_name))
      return

    print_io(module, saved_inputs, saved_outputs)

  def print_io(module, input, output):
    node = getattr(module, node_attr) if hasattr(module, node_attr) else None
    module_name = getattr(module, prefix_attr) if hasattr(module, prefix_attr) else None

    if node:
      print('node({}): {}'.format(node.idx, node.name))

    if module_name:
      print('module: {}({})'.format(module_name, type(module)))

    if isinstance(module, FakeQuantizer):
      print('quant_info:', module.quant_info())

    # saved_inputs/outputs may be empty tuple.
    print_blob(input, 'input')
    print_blob(output, 'output')
    print('')

  def print_blob(blob, prefix):
    if isinstance(blob, tuple) and len(blob) > 0:
      for idx, tensor in enumerate(blob):
        if isinstance(tensor, torch.Tensor):
          print('{}{}: {} {}'.format(prefix, idx, tensor.sum(), tensor.shape))
        else:
          print('{}{}: {}'.format(prefix, idx, tensor))
    elif isinstance(blob, torch.Tensor):
      print('{}: {} {}'.format(prefix, blob.sum(), blob.shape))
    else:
      print(prefix, None)

  def print_saved_blob(model):
    model.apply(print_saved_io)

  # Hook the node to module if the graph is provided.
  if graph:
    for node in graph.nodes:
      module = get_module_by_node(model, node)
      if module:
        setattr(module, node_attr, node)

  hook_func = save_io if mode == 'save' else print_io
  for name, module in model.named_modules():
    setattr(module, prefix_attr, name)
    module.register_forward_hook(hook_func)

  if mode == 'save':
    model.print_saved_blob = types.MethodType(print_saved_blob, model)
  return model

def enable_print_blob(model, graph=False):
  enable_dump_blob(model, graph, mode='print')

def enable_save_blob(model, graph=False):
  enable_dump_blob(model, graph, mode='save')
