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

from torch import nn

import numpy as np
import torch
import types

from nndct_shared.base import NNDCT_OP
from pytorch_nndct.utils import TorchGraphSymbol
from pytorch_nndct.utils import tensor_util

def module_name_from_node(node):
  """Extract module name from node.

  Examples:
    A node's name is
    'ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv1]/input.50',
    then extracted module name is 'layer4.1.conv1'
  """
  node_name = node if isinstance(node, str) else node.name
  if TorchGraphSymbol.NODE_NAME_SEPERATOR not in node_name:
    return None

  name_parts = []
  parts = node_name.split(TorchGraphSymbol.NODE_NAME_SEPERATOR)[1:-1]
  for part in parts:
    if '[' not in part or ']' not in part:
      raise RuntimeError(
          'Can not extract module name from node {}'.format(node_name))
    left_bracket = part.index('[')
    right_bracket = part.index(']')
    name_parts.append(part[left_bracket + 1:right_bracket])
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

def copy_module_weights(src_module, dst_module):

  if isinstance(src_module,
                (torch.nn.Conv2d, torch.nn.ConvTranspose2d)) or isinstance(
                    dst_module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
    if isinstance(src_module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
      dst_module.conv.weight.data.copy_(src_module.weight.data)
      if src_module.bias is not None:
        dst_module.conv.bias.data.copy_(src_module.bias.data)
    else:
      dst_module.weight.data.copy_(
          src_module.get_active_filter(dst_module.in_channels,
                                       dst_module.out_channels,
                                       dst_module.kernel_size).data)
      if dst_module.bias is not None:
        dst_module.bias.data.copy_(
            src_module.get_active_bias(dst_module.out_channels).data)

  elif isinstance(src_module, torch.nn.Linear) or isinstance(
      dst_module, torch.nn.Linear):

    if isinstance(src_module, torch.nn.Linear):
      dst_module.linear.weight.data.copy_(src_module.weight.data)
      if src_module.bias is not None:
        dst_module.linear.bias.data.copy_(src_module.bias.data)
    else:
      dst_module.weight.data.copy_(
          src_module.get_active_filter(dst_module.in_features,
                                       dst_module.out_features).data)
      if dst_module.bias is not None:
        dst_module.bias.data.copy_(
            src_module.get_active_bias(dst_module.out_features).data)

  elif isinstance(src_module, torch.nn.BatchNorm2d) or isinstance(
      dst_module, torch.nn.BatchNorm2d):
    if isinstance(src_module, torch.nn.BatchNorm2d):
      dst_module.bn.weight.data.copy_(
          src_module.weight.data[:dst_module.bn.num_features])
      dst_module.bn.bias.data.copy_(
          src_module.bias.data[:dst_module.bn.num_features])
      dst_module.bn.running_mean.data.copy_(
          src_module.running_mean.data[:dst_module.bn.num_features])
      dst_module.bn.running_var.data.copy_(
          src_module.running_var.data[:dst_module.bn.num_features])
    else:
      dst_module.weight.data.copy_(
          src_module.bn.weight.data[:dst_module.num_features])
      dst_module.bias.data.copy_(
          src_module.bn.bias.data[:dst_module.num_features])
      dst_module.running_mean.data.copy_(
          src_module.bn.running_mean.data[:dst_module.num_features])
      dst_module.running_var.data.copy_(
          src_module.bn.running_var.data[:dst_module.num_features])

def replace_modules(model, module_names, new_modules, copy_ckpt=False):
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
    if copy_ckpt:
      copy_module_weights(modules[i], new_modules[i])

def create_module_by_node(module_cls, node):
  attrs = {name: node.op.get_config(name) for name in node.op.configs}
  module = module_cls(**attrs)
  state_dict = state_dict_from_node(node)
  module_state_dict = {
      tensor_name.split('.')[-1]: tensor
      for tensor_name, tensor in state_dict.items()
  }
  module.load_state_dict(module_state_dict)
  return module

def state_dict_from_node(node):
  state_dict = {}
  # Copy from ModuleHooker::update_parameters in qproc/ModuleHooker.py
  for param_type, tensor in node.op.params.items():
    tensor_util.param_to_torch_format(tensor)
    data = np.copy(tensor.data)
    tensor_util.param_to_nndct_format(tensor)

    if node.op.type in [NNDCT_OP.CONVTRANSPOSE2D, NNDCT_OP.CONVTRANSPOSE3D
                       ] and param_type == node.op.ParamName.WEIGHTS:
      data = data.swapaxes(0, 1)
      data = np.ascontiguousarray(data)

    if node.op.type in [NNDCT_OP.DEPTHWISE_CONV2D, NNDCT_OP.DEPTHWISE_CONV3D
                       ] and param_type == node.op.ParamName.WEIGHTS:
      out_channels = node.node_config("out_channels")
      kernel_size = node.node_config("kernel_size")
      data = data.reshape((out_channels, 1, *kernel_size))

    if node.op.type in [
        NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D, NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D
    ] and param_type == node.op.ParamName.WEIGHTS:
      in_channels = node.node_config("in_channels")
      kernel_size = node.node_config("kernel_size")
      data = data.reshape((1, in_channels, *kernel_size))
      data = data.swapaxes(0, 1)
      data = np.ascontiguousarray(data)

    state_dict[tensor.name] = torch.from_numpy(data)
  return state_dict

def slim_model_from_state_dict(model, state_dict):
  """Modify modules according to their weights in the state dict and
  load the state dict to the model.

    Args:
      model: An torch.nn.Module instance to load state dict.
      state_dict: A state dict to be loaded.

    Returns:
      A modified model that matchs weights shape in the state dict.

  """
  for key, module in model.named_modules():
    weight_key = key + '.weight'
    bias_key = key + '.bias'
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
      module.weight = nn.Parameter(state_dict[weight_key])
      module.bias = nn.Parameter(state_dict[bias_key])
      module.running_mean = state_dict[key + '.running_mean']
      module.running_var = state_dict[key + '.running_var']
      module.num_features = module.weight.size(0)
    elif isinstance(module, (nn.Conv2d, nn.Conv3d)):
      assert module.groups == 1
      module.weight = nn.Parameter(state_dict[weight_key])
      if bias_key in state_dict:
        module.bias = nn.Parameter(state_dict[bias_key])
      module.out_channels = module.weight.size(0)
      module.in_channels = module.weight.size(1)
    elif isinstance(module, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
      assert module.groups == 1
      module.weight = nn.Parameter(state_dict[weight_key])
      if bias_key in state_dict:
        module.bias = nn.Parameter(state_dict[bias_key])
      module.in_channels = module.weight.size(0)
      module.out_channels = module.weight.size(1)
    elif isinstance(module, nn.Linear):
      module.weight = nn.Parameter(state_dict[weight_key])
      if bias_key in state_dict:
        module.bias = nn.Parameter(state_dict[bias_key])
      module.out_features = module.weight.size(0)
      module.in_features = module.weight.size(1)
    else:
      pass
  model.load_state_dict(state_dict)
  return model

def setattr_if_has(module, name, attr):
  if not hasattr(module, name):
    raise AttributeError(
        'Attempting to set an non-existing attribute "{}"'.format(name))
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
      print('[WARN] No saved blob: node={}, module={}\n'.format(
          node_name, module_name))
      return

    print_io(module, saved_inputs, saved_outputs)

  def print_io(module, input, output):
    node = getattr(module, node_attr) if hasattr(module, node_attr) else None
    module_name = getattr(module, prefix_attr) if hasattr(module,
                                                          prefix_attr) else None

    if node:
      print('node({}): {}'.format(node.idx, node.name))

    if module_name:
      print('module: {}({})'.format(module_name, type(module)))

    if hasattr(module, 'export_quant_info'):
      print('quant_info:', module.export_quant_info())

    # saved_inputs/outputs may be empty tuple.
    print_blob(input, 'input')
    print_blob(output, 'output')
    print('')

  def print_blob(blob, prefix):
    if isinstance(blob, tuple) and len(blob) > 0:
      for idx, tensor in enumerate(blob):
        if isinstance(tensor, torch.Tensor):
          print('{}[{}]: sum={}, dtype={}, shape={}'.format(
              prefix, idx, tensor.sum(), tensor.dtype, tensor.shape))
        else:
          print('{}{}: {}'.format(prefix, idx, tensor))
    elif isinstance(blob, torch.Tensor):
      print('{}: sum={}, dtype={}, shape={}'.format(prefix, blob.sum(),
                                                    blob.dtype, blob.shape))
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

def get_module_name(module):
  if isinstance(module, torch.jit.ScriptModule):
    return module.original_name

def visualize_tensors(model):
  from nndct_shared.utils import Plotter

  def plot_output_tensor(op, inputs, outputs):
    if isinstance(outputs, torch.Tensor):
      Plotter().plot_hist(op.node.name,
                          outputs.cpu().detach().numpy().flatten())

  for mod in model.modules():
    if mod is not model:
      mod.register_forward_hook(plot_output_tensor)

def get_flattened_input_args(input):

  def _flatten_args(input_args, flattened_inputs):
    if isinstance(input_args, (tuple, list)):
      for inp in input_args:
        _flatten_args(inp, flattened_inputs)
    else:
      flattened_inputs.append(input_args)

  flattened_inputs = []
  _flatten_args(input, flattened_inputs)
  return tuple(flattened_inputs)

def collect_input_devices(input_args):

  def _collect_device(inp, device_type_set):
    if isinstance(inp, torch.Tensor):
      device_type_set.add(inp.device.type)
    elif isinstance(inp, (list, tuple)):
      for i in inp:
        _collect_device(i, device_type_set)

  device_type_set = set()

  _collect_device(input_args, device_type_set)
  return device_type_set

def to_device(module, input_args, device):

  if input_args is not None:
    if isinstance(input_args, torch.Tensor):
      input_args = input_args.to(device)
    else:
      is_tuple = True if isinstance(input_args, tuple) else False
      input_args = list(input_args)
      for i in range(len(input_args)):
        _, inp = to_device(None, input_args[i], device)
        input_args[i] = inp
      if is_tuple:
        input_args = tuple(input_args)
  if module is not None:
    module = module.to(device)
  return module, input_args

def get_module_name(module):
  if isinstance(module, torch.jit.ScriptModule):
    return module.original_name
  else:
    return module._get_name()
