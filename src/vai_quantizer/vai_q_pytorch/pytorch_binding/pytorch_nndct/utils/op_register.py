

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

import functools
import inspect
import types
from typing import List, Optional

from nndct_shared.utils import GLOBAL_MAP, NNDCT_KEYS, NndctScreenLogger, NNDCT_OP, QError, QWarning, QNote 
import torch

from .nndct2torch_op_map import add_mapping_item
from .torch_const import TorchSymbol
from .torch_op_attr import gen_attr


def op_register(nndct_op: str, torch_op: str, force_to_primitive=False, schema=None, class_type=None):
  add_mapping_item(nndct_op, torch_op)
  gen_attr(torch_op, force_to_primitive=force_to_primitive, schema=schema, class_type=class_type)



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


def register_custom_op(op_type: str, attrs_list: Optional[List[str]] = None, mapping_to_xir=False):
  """The decorator is used to register the function as a custom operation.
  Args:
	op_type(str): The operator type registered into quantizer. 
                The type should not conflict with pytorch_nndct
                
	attrs_list(Optional[List[str]], optional): the name list of attributes that define operation flavor. 
                                              For example, Convolution operation has such attributes as padding, dilation, stride and groups. 
                                              The order of name in attrs_list should be consistent with that of the arguments list. 
                                              Default: None
  
  """
  def decorate(func):    
    if op_type in NNDCT_OP.__dict__.values() and (not mapping_to_xir):
      NndctScreenLogger().error2user(QError.OP_REGIST, f"'{op_type}' has been defined in pytorch_nndct, please use other type name.")     
      exit(1)                                                                                                                                                                                                                       
    if not inspect.isfunction(func):
      RuntimeError("This api only decorate a function object")
    
    custom_op_attr_map = GLOBAL_MAP.get_ele(NNDCT_KEYS.CUSTOM_OP_ATTRS_MAP)
    if custom_op_attr_map is None:
      custom_op_attr_map = {}
      GLOBAL_MAP.set_map(NNDCT_KEYS.CUSTOM_OP_ATTRS_MAP, custom_op_attr_map)
    if op_type in custom_op_attr_map:
      NndctScreenLogger().error2user(QError.OP_REGIST, f"'{op_type}' can't be registered multiple times.")
    else:
      custom_op_attr_map[op_type] = attrs_list if attrs_list is not None else []
      
    if mapping_to_xir is True:
      NndctScreenLogger().info(f'`{op_type}` has been mapped to xir.')
      custom2xir = GLOBAL_MAP.get_ele(NNDCT_KEYS.CUSTOM_TO_XIR_LIST)
      if custom2xir is None:
        custom2xir = []
        GLOBAL_MAP.set_map(NNDCT_KEYS.CUSTOM_TO_XIR_LIST, custom2xir)
      if op_type not in custom2xir:
        custom2xir.append(op_type) 
      else:
        raise RuntimeError(f"{op_type} has alrealy been mapped to XIR. Please use this op type instead of custom op.")
      
    @functools.wraps(func)
    def innner(*args, **kwargs):
      custom_op = types.new_class(op_type, (torch.autograd.Function,), {})
      custom_op.forward = staticmethod(func)
      return custom_op.apply(*args, **kwargs)
    return innner
  NndctScreenLogger().info(f'`{op_type}` has been registered as a custom op.')
  return decorate
