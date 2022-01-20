

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
from typing import Callable, Dict, Tuple

import torch
from nndct_shared.base import  NNDCT_KEYS, GLOBAL_MAP
from .torch_const import TorchOpClassType
from .jit_utils import find_builtin, modules_containing_builtins, builtin_ops
from .schema import SchemaHelper

_TORCH_OP_ATTR_MAP = {}  #Dict(str, TorchOpAttr)



class TorchOpAttr:

  def __init__(self, op_name: str):
    self.op_name = op_name
    self.op_class_type = None
    self.attrs = []  #List[str]
    self.input_args = []  #List[str]

  def set_op_class_type(self, force_to_primitive: bool, schema: "Schema", class_type=None):
    if class_type is not None:
      self.op_class_type = TorchOpClassType.CUSTOM_FUNCTION
    elif schema is not None:
      schema2torchop = GLOBAL_MAP.get_ele(NNDCT_KEYS.TORCH_SCHEMA_OP_TABLE)
      schema_handler = SchemaHelper(schema)
      torchop = schema2torchop[schema_handler.toString()]
      self.op_class_type = torchop.op_class_type
    else:
      if force_to_primitive:
        self.op_class_type = TorchOpClassType.PRIMITIVE
      else:
        if self.op_name in dir(torch.nn):
          self.op_class_type = TorchOpClassType.NN_MODULE
          self.op_name = '.'.join(['torch', 'nn', self.op_name])

        elif self.op_name in dir(torch.nn.functional):
          self.op_class_type = TorchOpClassType.NN_FUNCTION
          self.op_name = '.'.join(['torch', 'nn', 'functional', self.op_name])

        elif self.op_name in dir(torch) and isinstance(getattr(torch, self.op_name), Callable):
          self.op_class_type = TorchOpClassType.TORCH_FUNCTION
          self.op_name = '.'.join(['torch', self.op_name])

        elif self.op_name in dir(torch.Tensor):
          self.op_class_type = TorchOpClassType.TENSOR

        else:
          self.op_class_type = TorchOpClassType.UNKNOWN

  def fill_in_attr_and_inputs_table(self, schema):
    if schema is not None and self.op_class_type == TorchOpClassType.NN_FUNCTION:
      schema_handler = SchemaHelper(schema)
      for arg in schema_handler.get_arguments():
        if schema_handler.arg_name(arg) == "self":
          self.attrs.append("input")
        else:
           self.attrs.append(schema_handler.arg_name(arg))
        
    else:
      if self.op_class_type == TorchOpClassType.NN_MODULE:
        self.attrs[:] = list(
            eval('inspect.signature({}.__init__).parameters'.format(
                self.op_name)))[1:]
        self.input_args[:] = list(
            eval('inspect.signature({}.forward).parameters'.format(
                self.op_name)))[1:]

      elif self.op_class_type == TorchOpClassType.NN_FUNCTION:
        self.attrs[:] = list(
            eval('inspect.signature({}).parameters'.format(self.op_name)))[:]
        self.input_args[:] = self.attrs[:]

def gen_attr(torch_op: str, force_to_primitive: bool, schema: "Schema", class_type=None):
  global _TORCH_OP_ATTR_MAP
  if torch_op not in _TORCH_OP_ATTR_MAP:
    op_attr = TorchOpAttr(torch_op)
    op_attr.set_op_class_type(force_to_primitive, schema, class_type)
    op_attr.fill_in_attr_and_inputs_table(schema)
    _TORCH_OP_ATTR_MAP[torch_op] = op_attr

def get_torch_op_attr_map():
  global _TORCH_OP_ATTR_MAP
  if len(_TORCH_OP_ATTR_MAP) == 0:
    raise Exception(
        'please build the torch_op_type - > torch_op_attributes map')
  return _TORCH_OP_ATTR_MAP

def get_torch_op_attr(torch_op_type):
  if get_torch_op_attr_map().get(torch_op_type, None) is None:
    raise Exception(
        'pleas check torch op attribute :"{}"'.format(torch_op_type))
  else:
    return get_torch_op_attr_map()[torch_op_type]



from functools import namedtuple
TorchOp = namedtuple("TorchOp", ["name", "caller", "op_class_type"])


def _hidden(name):
    return name.startswith('_') and not name.startswith('__')


def _make_pair(schema, torchop):
  schema_op_dict = GLOBAL_MAP.get_ele(NNDCT_KEYS.TORCH_SCHEMA_OP_TABLE)
  schema_handler = SchemaHelper(schema)
  if schema_handler.toString() not in schema_op_dict:
    schema_op_dict[schema_handler.toString()] = torchop
  
  
def _get_tensor_ops():
  def is_tensor_method(schema):
    if len(schema.arguments) == 0:
        return False
    self = schema.arguments[0]
    if self.name != 'self':
        return False
    if not self.type.isSubtypeOf(torch._C.TensorType.get()):
        return False
    return True

  # discover methods
  for elem in dir(torch.Tensor):
      if not _hidden(elem):
        schemas = torch._C._jit_get_schemas_for_operator("aten::" + elem)
        for schema in schemas:
          if is_tensor_method(schema):
            # aten_schema = SchemaWrapper(f"aten::{elem}", schema)
            torchop = TorchOp(name=elem, caller=None, op_class_type=TorchOpClassType.TENSOR)
            _make_pair(schema, torchop)


      
def _get_nn_functional_ops():

    # Iterate over torch.nn.functional
    mod = torch.nn.functional
    name = mod.__name__
    for elem in dir(torch.nn.functional):
        attr = getattr(mod, elem)
        if not inspect.isfunction(attr) or _hidden(elem[0]):
            # Ignore non-functions and internal methods
            continue

        attr_module = inspect.getmodule(attr)
        if not attr_module:
            raise RuntimeError(f'Module for {attr} not found')

        if 'torch.nn.functional' not in attr_module.__name__:
            # Ignore functions from outside torch.nn.functional
            continue

        try:
            # compile fn, get schema
            scripted = torch.jit.script(attr)
            schema = scripted.schema
            torchop = TorchOp(name=elem, caller=attr, op_class_type=TorchOpClassType.NN_FUNCTION)
            _make_pair(schema, torchop)
        except:  # noqa
            # Skip interpolate / boolean dispatched things
            pass

    # Iterate over modules that we know contain a lot of builtins
    for mod in modules_containing_builtins():
      name = mod.__name__
      for elem in dir(mod):
        builtin = find_builtin(getattr(mod, elem))
        if builtin is not None:
    
          schemas = torch._C._jit_get_schemas_for_operator(builtin)
          for schema in schemas:
            # remove _tan but not __and__
            if not _hidden(elem):
              # aten_schema = SchemaWrapper(f"{builtin}", schema)
              if name == "torch._C._nn":
                torchop = TorchOp(name=elem, caller=getattr(mod, elem), op_class_type=TorchOpClassType.NN_CORE_FUNCTION)
              else:
                torchop = TorchOp(name=elem, caller=getattr(mod, elem), op_class_type=TorchOpClassType.TORCH_FUNCTION)
              _make_pair(schema, torchop) 


def _get_builtins_helper():
  builtins = []
  for fn, _builtin_name in builtin_ops():
    mod = inspect.getmodule(fn)

    if not hasattr(fn, '__name__'):
        # typing classes
        continue
    if not mod:
        continue
    if _hidden(fn.__name__) or _hidden(fn.__qualname__) or _hidden(mod.__name__):
        # skip internal-only methods
        continue

    if 'torch._C' in mod.__name__:
        continue

    builtins.append((fn, _builtin_name))

  return builtins
  
  
def _is_math_fn(fn):
  mod = inspect.getmodule(fn)
  if not mod:
      raise RuntimeError(f'Module for {fn} not found')

  return mod.__name__ == 'math'
  
def _get_torchscript_builtins():
    builtins = filter(lambda fn: not _is_math_fn(fn[0]), _get_builtins_helper())
    builtins_list = list(builtins)
    # Iterate over the specially added builtins
    for fn, _builtin_name in builtins_list:
      mod = inspect.getmodule(fn)
      if not mod:
        raise RuntimeError(f'Module for {fn} not found')
      builtin = find_builtin(fn)
      if builtin is not None:
        schemas = torch._C._jit_get_schemas_for_operator(builtin)
        for schema in schemas:
          # aten_schema = SchemaWrapper(f"{builtin}", schema)
          torchop = TorchOp(name=builtin, caller=fn, op_class_type=TorchOpClassType.TORCH_SCRIPT_BUILTIN_FUNCTION)
          _make_pair(schema, torchop)  

def _get_math_builtins():
  builtins = filter(lambda fn: _is_math_fn(fn[0]), _get_builtins_helper())
  builtins_list = list(builtins)
  # Iterate over the specially added builtins
  for fn, _builtin_name in builtins_list:
      mod = inspect.getmodule(fn)
      if not mod:
          raise RuntimeError(f'Module for {fn} not found')
      builtin = find_builtin(fn)
      if builtin is not None:
          schemas = torch._C._jit_get_schemas_for_operator(builtin)
          for schema in schemas:
              # aten_schema = SchemaWrapper(f"{builtin}", schema)
              schema_str = SchemaHelper(schema).toString()
              if 'Tensor' in schema_str:
                  # Skip Tensor ops that have the same name as math functions
                  # (they will show up in the tensor methods section)
                  continue
              torchop = TorchOp(name=builtin, caller=fn, op_class_type=TorchOpClassType.MATH_BUILTIN_FUNCTION)
              _make_pair(schema, torchop)
              


def _get_global_builtins():
    # Taken from the 'globals' map in torch/csrc/jit/frontend/ir_emitter.cppsss
  supported_builtins = [
        'print',
        'tuple',
        'float',
        'int',
        'bool',
        'str',
        'getattr',
        'hasattr',
        'isinstance',
        'len',
        'hex',
        'oct',
        'round',
        'hash',
        'min',
        'max',
        'abs',
        'all',
        'divmod',
        'list',
        'ord',
        'chr',
        'bin',
        'range',
        'zip',
        'enumerate',
        'sorted',
    ]

  op_renames = {
      'bool': 'aten::Bool',
      'int': 'aten::Int',
      'float': 'aten::Float',
      'abs': 'prim::abs',
      'max': 'prim::max',
      'min': 'prim::min',
      'range': 'fake::does_not_exist',
  }

  for fn in supported_builtins:
    op_name = 'aten::{}'.format(fn)
    if fn in op_renames:
        op_name = op_renames[fn]
    schemas = torch._C._jit_get_schemas_for_operator(op_name)
    for s in schemas:
      # aten_schema = SchemaWrapper(f"{op_name}", s)
      torchop = TorchOp(name=op_name, caller=__builtins__[fn], op_class_type=TorchOpClassType.GLOBAL_BUILTIN_FUNCTION)
      _make_pair(s, torchop)
      
        
    
def build_aten_torch_ops_table():
  op_gathering_fns = (_get_tensor_ops, 
                      _get_nn_functional_ops, 
                      _get_torchscript_builtins, 
                      _get_global_builtins, 
                      _get_math_builtins,
                      )
  schema2torchop = GLOBAL_MAP.get_ele(NNDCT_KEYS.TORCH_SCHEMA_OP_TABLE)
  # schema_lut = GLOBAL_MAP.get_ele(NNDCT_KEYS.SCHEMA_LUT)
  if not schema2torchop:
    
    schema2torchop: Dict[str, TorchOp] = {}
    GLOBAL_MAP.set_map(NNDCT_KEYS.TORCH_SCHEMA_OP_TABLE, schema2torchop)

    # schema_lut: Dict[Tuple(str, int), "Schema"] = {}
    for fn in op_gathering_fns:
      fn()



    # for key, value in schema2torchop.items():
    #   schema_str = SchemaHelper(key).toString()
    #   print(key.name, schema_str)
    #   parsed_schema = torch._C.parse_schema(schema_str)
    #   if SchemaHelper(parsed_schema).toString() != schema_str:
    #     print(f"parsed_schema:{SchemaHelper(parsed_schema).toString()}")
    #     print(f"schema_str:{schema_str}")
    #     assert False
