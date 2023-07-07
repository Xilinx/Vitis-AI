

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

from nndct_shared.nndct_graph.base_tensor import Tensor
from pytorch_nndct.utils import TorchGraphSymbol
_GRAPH_SCOPE_SYM = TorchGraphSymbol.GRAPH_SCOPE_SYM

# IGNORE_STATEDICT_KEYS = ['num_batches_tracked']
scalar_type_to_pytorch_type = [
    'torch.uint8',        # 0
    'torch.int8',         # 1
    'torch.short',        # 2
    'torch.int',          # 3
    'torch.int64',        # 4
    'torch.half',         # 5
    'torch.float',        # 6
    'torch.double',       # 7
    'torch.complex32',    # 8
    'torch.complex64',    # 9
    'torch.complex128',   # 10
    'torch.bool',         # 11
    'torch.qint8',        # 12
    'torch.quint8',       # 13
    'torch.qint32',       # 14
    'torch.bfloat16',     # 15
]

def convert_np_type_to_pytorch_type(np_type):
  return {
  'int64': 'torch.int64',
  'int32': 'torch.int32',
  'float32': 'torch.float',
  'float64': 'torch.double'
  }.get(np_type, np_type)


def convert_dtype_between_np_and_pytorch(dtype):
  return {
  'int64': 'torch.int64',
  'int32': 'torch.int32',
  'float32': 'torch.float',
  'float64': 'torch.double',
  'torch.int64': 'int64',
  'torch.long': 'int64',
  'torch.int32': 'int32',
  'torch.int': 'int32',
  'torch.float32': 'float32',
  'torch.float': 'float32',
  'torch.float64': 'float64',
  'torch.double': 'float64',
  }.get(dtype, dtype)


def get_full_name(graph_name: str, name: str) -> str:
  """get the full name of node/tensor in graph

  Args:
     graph_name (str): graph name

  Returns:
     str: full name
  """

  return _GRAPH_SCOPE_SYM.join([graph_name, name])


def get_short_name(full_name: str) -> str:
  """get the name of node/tensor in graph without graph name

  Args:
      full_name (str): full name of node/tensor
  Returns:
      str: short name
  """
  return full_name.split(_GRAPH_SCOPE_SYM)[-1]


def get_formal_name(hier_name: str) -> str:
  """replace `.` with `_`
  Args:
      hier_name (str): "layer_0.layer_1"

  Returns:
      str: "layer_0_layer_1"
  """
  return get_short_name(hier_name.replace(".", "_"))


def create_graph_handler(module):
  import torch
  if isinstance(module, torch.jit.ScriptModule):
    from .script_helper import TorchScriptModuleHandler
    return TorchScriptModuleHandler()

  elif isinstance(module, torch.nn.Module):
    from .trace_helper import TorchGraphHandler
    return TorchGraphHandler()
  else:
    raise NotImplementedError()


def python_dtype(value):
  type_map = {
    "torch.int": "int",
    "torch.long": "int",
    "torch.short": "int",
    "torch.float": "float",
    "torch.half": "float",
    "torch.double": "float",
    "torch.bool": "bool",
    int: "int",
    float: "float",
    bool: "bool",
    str: "str",
    "int64": "int",
    "int32": "int",
    "float32": "float",
    "float64": "float",
    "float16": "float"
  }
  if isinstance(value, Tensor):
    return type_map.get(value.dtype, value.dtype)
  else:
    return type_map[type(value)]


class TorchDeviceType(object):
  CUDA = "cuda"
  CPU = "cpu"
  UNKOWN = "unkown"



class ValueDeviceInfo(object):
  def __init__(self, torch_device="unkown"):
    self._device_map = {
      "cpu": TorchDeviceType.CPU,
      "cuda": TorchDeviceType.CUDA,
      "unkown": TorchDeviceType.UNKOWN
    }
    if hasattr(torch_device, "type"):
      self._type = self._device_map.get(torch_device.type, TorchDeviceType.UNKOWN)
    else:
      self._type = self._device_map.get(torch_device, TorchDeviceType.UNKOWN)
  
  @property
  def device_type(self):
    return self._type 

