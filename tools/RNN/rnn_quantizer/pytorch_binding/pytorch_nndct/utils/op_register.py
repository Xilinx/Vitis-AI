

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

import inspect
import functools
from functools import wraps
from .nndct2torch_op_map import add_mapping_item
from .torch_op_attr import gen_attr
from .torch_const import TorchSymbol

def op_register(nndct_op: str, torch_op: str, force_to_primitive=False, schema=None):
  add_mapping_item(nndct_op, torch_op)
  gen_attr(torch_op, force_to_primitive=force_to_primitive, schema=schema)



_QUANT_MODULES = []  #List[str]

def register_quant_op(func):
  if not inspect.isfunction(func):
    raise RuntimeError("Only decorate function")
  global _QUANT_MODULES
  _QUANT_MODULES.append(func.__name__)
  @functools.wraps(func)
  def innner(*args, **kwargs):
    return func(*args, **kwargs)
  return innner
    
    

def get_defined_quant_module(torch_op: str):
  quant_modules_lower = [module.lower() for module in _QUANT_MODULES]
  if torch_op.lower() in quant_modules_lower:
    index = quant_modules_lower.index(torch_op.lower())
    return ".".join([TorchSymbol.MODULE_PREFIX, _QUANT_MODULES[index]])



    
    
    