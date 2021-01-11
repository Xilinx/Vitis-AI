

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
import importlib
import random
import string
import sys
import tempfile
import torch

from collections import OrderedDict
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from nndct_shared.base import NNDCT_OP
from nndct_shared.nndct_graph.base_tensor import Tensor
from nndct_shared.pruning import pruning_lib
from nndct_shared.utils import logging
from pytorch_nndct.qproc import utils as api_utils
from pytorch_nndct.export import get_script_writer
from pytorch_nndct.utils import TorchSymbol

_torch_layouts = {2: 'OI', 4: 'OIHW'}
_nndct_layouts = {2: 'OI', 4: 'OHWI'}

def torch_to_nndct(tensor):
  return transpose_tensor(tensor, _torch_layouts, _nndct_layouts)

def nndct_to_torch(tensor):
  return transpose_tensor(tensor, _nndct_layouts, _torch_layouts)

def transpose_tensor(tensor, src_layouts, dst_layouts):
  if not isinstance(tensor, Tensor):
    raise TypeError("'tensor' must be Tensor, but given {}".format(
        type(tensor)))
  if tensor.ndim != 4 and tensor.ndim != 2:
    return tensor

  src_layout = src_layouts[tensor.ndim]
  dst_layout = dst_layouts[tensor.ndim]

  axis = [src_layout.index(d) for d in dst_layout]
  tensor.transpose(axis)
  return tensor

def torch_tensor_from_nndct(tensor):
  replicated_tensor = copy.deepcopy(tensor)
  return torch.from_numpy(
      transpose_tensor(replicated_tensor, _nndct_layouts, _torch_layouts).data)

def _inspect_ana(sens_path):
  net_sens = pruning_lib.read_sens(sens_path)
  print(net_sens)
  return net_sens

def dummy_inputs(input_specs):
  inputs = []
  for spec in input_specs:
    inputs.append(torch.rand(1, *spec.shape).type(spec.dtype))
  return inputs

def unwrap_parallel_module(module):
  if isinstance(module, (DataParallel, DistributedDataParallel)):
    model = module.module
  else:
    model = module
  return model

def raw_param_name(full_param_name):
  return full_param_name.split('.')[-1]

def random_str(str_length=4):
  letters = string.ascii_lowercase
  return ''.join(random.choice(letters) for i in range(str_length))

def readable_num(number):
  s = ''
  if number < 0:
    s += '-'
    number = -number

  if number < 1000:
    s += '%d' % number
  elif number > 1e15:
    s += '%0.3G' % number
  else:
    units = 'KMGT'
    unit_index = 0
    while number > 1000000:
      number /= 1000
      unit_index += 1
    s += '%.2f%s' % (number / 1000.0, units[unit_index])
  return s

def pad_to_sparse_tensor(tensor, pruning_info):
  """Pad tensor with zeros by given pruning_info.
    Restore the tensor to its original unpruned shape and use zeros to fill
    in the removed input/output channels.
    [100, 60, 3, 3] -> [128, 64, 3, 3]
  """
  shape = list(tensor.shape)
  # OIHW for 4 dims, OI for 2 dims.
  orig_out_channels = shape[0] + len(pruning_info.removed_outputs)

  # Pad output channels.
  shape[0] = orig_out_channels
  out_padded_tensor = torch.zeros(shape, dtype=tensor.dtype)
  index = 0
  for axis in range(orig_out_channels):
    if axis not in pruning_info.removed_outputs:
      out_padded_tensor[axis] = tensor[index]
      index += 1

  if len(shape) < 2:
    return out_padded_tensor

  orig_in_channels = shape[1] + len(pruning_info.removed_inputs)
  # Pad input channels.
  shape[1] = orig_in_channels
  in_padded_tensor = torch.zeros(shape, dtype=tensor.dtype)
  index = 0
  for axis in range(orig_in_channels):
    if axis not in pruning_info.removed_inputs:
      in_padded_tensor[:, axis] = out_padded_tensor[:, index]
      index += 1
  return in_padded_tensor

def rebuild_module(graph):
  _, filename = tempfile.mkstemp(suffix='.py', text=True)
  writer = get_script_writer(enable_quant=False)
  writer.write(graph, filename)

  #module_name = graph.name
  py_module_name = "_".join(["nndct", random_str()])
  spec = importlib.util.spec_from_file_location(py_module_name, filename)
  py_module = importlib.util.module_from_spec(spec)
  sys.modules[py_module_name] = py_module
  spec.loader.exec_module(py_module)
  rebuilt_module = py_module.__dict__[graph.name]()

  api_utils.connect_module_with_graph(rebuilt_module, graph)
  return rebuilt_module, filename

def map_rebuilt_module_to_node(model, graph):
  module_to_node = {}
  for name, module in model.named_children():
    # module_name -> node_id
    node_idx = int(name.split(TorchSymbol.MODULE_NAME_SEPERATOR)[-1])
    node = graph.get_node_by_idx(node_idx)
    module_to_node[name] = node
  return module_to_node

def map_original_module_to_node(model, graph):
  module_to_node = {}
  for node in graph.nodes:
    attr_names = []
    # TODO(yuwang): Use pytorch_nndct/utils/module_utils.py::module_name_from_node
    # ResNet/Sequential[layer4]/BasicBlock[1]/Conv2d[conv1]/input.50
    parts = node.name.split('/')[1:-1]
    for part in parts:
      left_bracket = part.index('[')
      right_bracket = part.index(']')
      attr_names.append(part[left_bracket + 1:right_bracket])

    module = model
    for attr_name in attr_names:
      module = getattr(module, attr_name)
    module_to_node[id(module)] = node.name
  return module_to_node
