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
import importlib
import os
import random
import string
import sys
import tempfile
import torch

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from nndct_shared.nndct_graph.base_tensor import Tensor
from nndct_shared.pruning import errors
from pytorch_nndct import parse
from pytorch_nndct.export import get_script_writer
from pytorch_nndct.parse import parse_utils
from pytorch_nndct.qproc import utils as api_utils
from pytorch_nndct.utils import TorchSymbol
from pytorch_nndct.utils import module_util as mod_util

_torch_layouts = {2: 'OI', 4: 'OIHW'}
_nndct_layouts = {2: 'OI', 4: 'OHWI'}

def torch_to_nndct(tensor):
  return transpose_tensor(tensor, _torch_layouts, _nndct_layouts)

def nndct_to_torch(tensor):
  return transpose_tensor(tensor, _nndct_layouts, _torch_layouts)

def transpose_tensor(tensor, src_layouts, dst_layouts):
  if not isinstance(tensor, Tensor):
    raise errors.OptimizerDataFormatError(
        "'tensor' must be Tensor, but given {}".format(type(tensor)))
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

def dummy_inputs(input_specs):
  inputs = []
  for spec in input_specs:
    inputs.append(torch.rand(*spec.shape).type(spec.dtype).cuda())
  return tuple(inputs)

def unwrap_parallel_module(module):
  if isinstance(module, (DataParallel, DistributedDataParallel)):
    model = module.module
  else:
    model = module
  return model

def random_str(str_length=4):
  letters = string.ascii_lowercase
  return ''.join(random.choice(letters) for i in range(str_length))

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

  # Pad input channels.
  orig_in_channels = shape[1] + len(pruning_info.removed_inputs)
  shape[1] = orig_in_channels
  in_padded_tensor = torch.zeros(shape, dtype=tensor.dtype)
  index = 0
  for axis in range(orig_in_channels):
    if axis not in pruning_info.removed_inputs:
      in_padded_tensor[:, axis] = out_padded_tensor[:, index]
      index += 1
  return in_padded_tensor

def is_debug_mode():
  return os.environ.get('VAI_OPTIMIZER_DEBUG', None) == '1'

def parse_to_graph(module, inputs, debug=False):
  if debug:
    from nndct_shared.utils import NndctOption, NndctDebugLogger
    NndctDebugLogger("vai_opt_debug.log")
    NndctOption.nndct_parse_debug.value = 5

  parser = parse.TorchParser()
  graph = parser(module._get_name(), module, inputs)

  if debug:
    from nndct_shared.utils import saving
    saving.save_graph(graph, hdf5_path='{}.hdf5'.format(graph.name))
    write_graph_script(graph, '{}_baseline.py'.format(graph.name))
  return graph

def write_graph_script(graph, filename=None):
  if not filename:
    _, filename = tempfile.mkstemp(suffix='.py', text=True)
  writer = get_script_writer(enable_quant=False)
  writer.write(graph, filename)
  return filename

def rebuild_model(graph, filename=None):
  filename = write_graph_script(graph, filename)

  #module_name = graph.name
  py_module_name = "_".join(["nndct", random_str()])
  spec = importlib.util.spec_from_file_location(py_module_name, filename)
  py_module = importlib.util.module_from_spec(spec)
  sys.modules[py_module_name] = py_module
  spec.loader.exec_module(py_module)
  rebuilt_model = py_module.__dict__[graph.name]()

  api_utils.connect_module_with_graph(rebuilt_model, graph)
  return rebuilt_model, filename

def map_rebuilt_module_to_node(model, graph):
  module_to_node = {}
  for name, module in model.named_children():
    # module_name -> node_id
    idx = int(name.split(TorchSymbol.MODULE_NAME_SEPERATOR)[-1])
    node = graph.get_node_by_idx(idx)
    module_to_node[name] = node
  return module_to_node

def map_original_module_to_node(model, graph):
  module_to_node = {}
  for node in graph.nodes:
    module = mod_util.get_module_by_node(model, node)
    if module:
      module_to_node[id(module)] = node.name
  return module_to_node

def excluded_node_names(model, graph, excludes):
  excluded_nodes = []
  module_to_node = map_original_module_to_node(model, graph)
  for exclude in excludes:
    if isinstance(exclude, str):
      excluded_nodes.append(exclude)
    elif isinstance(exclude, torch.nn.Module):
      for module in exclude.modules():
        module_id = id(module)
        if module_id in module_to_node:
          excluded_nodes.append(module_to_node[module_id])
    else:
      raise errors.OptimizerInvalidArgumentError(
          'Excludes must be either string or torch.nn.Module')
  return excluded_nodes

def state_dict_key_from_tensor(tensor):
  return tensor.name.split(parse_utils._GRAPH_SCOPE_SYM)[-1]

def get_actual_device(gpu: int) -> int:
  gpu = int(gpu)
  if 'CUDA_VISIBLE_DEVICES' in os.environ:
    available_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(",")
    if gpu >= len(available_devices):
      raise ValueError(
          f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}, arg gpu must be less than {len(available_devices)}"
      )
    return int(available_devices[gpu])
  else:
    return gpu
