
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

import json
import numpy as np
from collections import OrderedDict
from enum import Enum, auto
from functools import partial
from typing import Dict, List, Callable, Optional, Union, Any, Set

from nndct_shared.nndct_graph.base_tensor import Tensor
from nndct_shared.utils.common import AutoName

class AttrUser(object):
  def __init__(self, user, attr_name):
    self._user = user
    self._attr_name = attr_name

  @property
  def user(self):
    return self._user

  @user.setter
  def user(self, user):
    self._user = user

  @property
  def attr_name(self):
    return self._attr_name

  @attr_name.setter
  def attr_name(self, attr_name):
    self._attr_name = attr_name

def _default_read_and_write_value(value_mem: List[Any],
                                  in_out: int,
                                  attr_value: Optional[Any] = None):
  r""" if in_out == 0 stamp value in memory, otherwise get memory_value"""
  if in_out == 0:
    if isinstance(attr_value, (list, tuple, set)):
      value_mem[:] = list(attr_value)
    else:
      value_mem[:] = [attr_value]
  else:
    #if len(value_mem) == 1:
    if len(value_mem) == 1 and (not isinstance(value_mem[0], (tuple, list))):
      return value_mem[0]
    else:
      return value_mem[:]

class OccurenceType(Enum):
  REQUIRED = auto()
  OPTIONAL = auto()

class NndctIrAttr(object):
  r"""
  data structure used to hold nndct op attributes

  Args:
    name:  the attr name defined in op
    value_type : the type of element in attr value, such as 'int, float, bool...'

    size: the length of attr value. if size=None, that means the attr value has varible length

    occurence_type: mark the attr whether needs to be assigned

    default_value: need to be given when the occurence_type == OPTIONAL

    annotation: string description of this attr

    read_and_write_value_func: the function used to read value or write value to attr value chunk memory,
                              if it is not given, will use default function which just simply copy value
                              into memory without adjustment.

  """

  def __init__(self,
               name: AutoName,
               value_type: Any,
               size: Union[int, None],
               value_mem: List[Any],
               occurence_type: OccurenceType,
               default_value: Optional[List[Any]] = None,
               annotation: Optional[str] = None,
               map_to_xir: bool = True,
               read_and_write_value_func: Optional[Callable] = None) -> None:
    self._name = name
    self._type = value_type
    self._size = size
    self._value_mem = value_mem
    self._occurence_type = occurence_type
    self._annotation = annotation
    self._is_xir_attr = map_to_xir
    self._is_container = None
    if read_and_write_value_func is not None:
      self._read_and_write_value_func = read_and_write_value_func
    elif self._value_mem is not None:
      self._read_and_write_value_func = partial(
          _default_read_and_write_value, value_mem=self._value_mem)
    else:
      raise RuntimeError(
          "please provide read_and_write_value_fun or value memory for NndctIrAttr"
      )

    if self._occurence_type == OccurenceType.OPTIONAL:
      self.value = default_value

  @classmethod
  def create_attr_from(cls, attr_value):
    return cls(attr_value._name,
               attr_value._type,
               attr_value._size,
               list(attr_value._value_mem),
               attr_value._occurence_type,
               attr_value.value,
               attr_value._annotation,
               attr_value._is_xir_attr)





  @property
  def type(self):
    return self._type

  @property
  def value(self) -> Any:
    ret = self._read_and_write_value_func(in_out=1)
    if self._is_container and (not isinstance(ret, (list, tuple))):
      ret = [ret]
    return ret

  @value.setter
  def value(self, value):
    if not isinstance(value, (list, tuple, set)):
      value = [value]
      self._is_container = False
    else:
      value = list(value)
      self._is_container = True
    if value:
      if self._type is not Any and (not isinstance(value[0], (self._type, type(None)))):
        raise TypeError(
          f"The type of attr '{self._name.value}' should be {self._type} instead of {type(value[0])}"
        )
      if self._size is not None and len(value) != self._size:
        raise ValueError(
          f"the length of  value of {self._name.value} is not equal to {self._size}"
        )
    self._read_and_write_value_func(in_out=0, attr_value=value)

  @property
  def is_xir_attr(self):
    return self._is_xir_attr

class AttrParamIndexer(object):
  def __init__(self, getter, setter, enum_cls):
    self._getter = getter
    self._setter = setter
    self._enum_cls = enum_cls

  def _key_to_enum(self, key):
    for enum in self._enum_cls:
      if key == enum.name.lower():
        return enum
    raise KeyError('Not a valid key: {}'.format(key))

  def __getitem__(self, key):
    enum = self._key_to_enum(key)
    return self._getter(enum)

  def __setitem__(self, key, value):
    enum = self._key_to_enum(key)
    self._setter(enum, value)

class Operation(object):
  """Object that takes one or more tensors as input and outputs tensors."""

  def __init__(self, optype: str):
    self._attrs: Dict[Enum, NndctIrAttr] = {}
    # Attr names defined in the framework.
    self._configs: List[str] = []
    self._params: Dict[Enum, Tensor] = OrderedDict()
    self._type = optype
    self._is_custom_op = False
    self._export_attr_and_param()


  def clone_from(self, src_op, local_map):
    for attr_name, attr_value in src_op._attrs.items():
      new_attr_value = NndctIrAttr.create_attr_from(attr_value)
      new_attr_value.value = self.clone_attr_from(attr_name, attr_value.value, local_map)
      self._attr_value_mem[attr_name] = new_attr_value._value_mem
      self._attrs[attr_name] = new_attr_value

    for i, config in enumerate(src_op.configs):
      value = src_op.get_config(config)
      new_value = self.clone_attr_from(config, value, local_map)
      self._configs[i] = config
      setattr(self, config, new_value)

    for param_name, param_tensor in src_op.params.items():
      if param_tensor.name in local_map:
        self._params[param_name] = local_map[param_tensor.name]
      else:
        tensor = Tensor(name=param_tensor.name)
        tensor.clone_from(param_tensor)
        local_map[param_tensor.name] = tensor
        self._params[param_name] = tensor

    self._is_custom_op = src_op._is_custom_op


  def clone_attr_from(self, attr_name, attr_value, local_map):

    def _replace_attr_value(value_mem):

      if isinstance(value_mem, list):
        new_value = []
        for val in value_mem:
          new_value.append(_replace_attr_value(val))
        return new_value

      if isinstance(value_mem, Tensor):
        tensor = local_map[value_mem.name]
        attr_use = AttrUser(self, attr_name)
        tensor._attr_uses.append(attr_use)
        return tensor
      else:
        return value_mem

    new_value = _replace_attr_value(attr_value)
    return new_value


  def _define_attr(self,
                   name,
                   value_type,
                   size,
                   value_mem=None,
                   required=True,
                   default_value=None,
                   annotation=None,
                   read_and_write_value_func=None):
    if not hasattr(self, '_attr_value_mem'):
      self._attr_value_mem = {}
    if name not in self._attr_value_mem:
      self._attr_value_mem[name] = [None]

    if not value_mem:
      value_mem = self._attr_value_mem[name]

    occurence_type = OccurenceType.REQUIRED if required else OccurenceType.OPTIONAL

    self._attrs[name] = NndctIrAttr(
        name=name,
        value_type=value_type,
        size=size,
        value_mem=value_mem,
        occurence_type=occurence_type,
        annotation=annotation,
        read_and_write_value_func=read_and_write_value_func)

  def _export_attr_and_param(self):
    """Wrappers for self._attrs and self._params."""
    # Get/set attr/param by indexing.
    # e.g. op.attr['out_dim'] = 32
    #      op.param['weights']
    if hasattr(self, 'AttrName'):
      self.attr = AttrParamIndexer(
          self.get_attr, self.set_attr, self.AttrName)

    if hasattr(self, 'ParamName'):
      self.param = AttrParamIndexer(
          self.get_param, self.set_param, self.ParamName)

  def __repr__(self):
    return json.dumps(self.description(), indent=2, separators=(',', ': '))

  def description(self):

    def serialize_attr(attr_value):
      if isinstance(attr_value, (tuple, list)):
        attr_values = attr_value
      else:
        attr_values = [attr_value]

      serialized_values = []
      for value in attr_values:
        serialized_values.append(serialize_value(value))

      if not isinstance(attr_value, (tuple, list)):
        return serialized_values[0]
      elif isinstance(attr_value, tuple):
        return tuple(serialized_values)
      else:
        return serialized_values

    def serialize_value(value):
      if isinstance(value, Operation):
        return value.__class__.__name__
      elif isinstance(value, type):
        return value.__name__
      elif isinstance(value, Tensor):
        return {'value': value.name, 'type': value.__class__.__name__}
      else:
        return {'value': value, 'type': value.__class__.__name__}

    op_des = {}
    op_des['type'] = self._type
    op_des['param'] = {}
    for name, tensor in self._params.items():
      # In tf.keras, multiple rnn layers have the same weight name and
      # cannot be mapped to ParamName, so use string as the key to store
      # these rnn weights.

      if not isinstance(name, str):
        name = name.value
      if isinstance(tensor, list):
          op_des['param'][name] = [item.name for item in tensor]
      else:
          op_des['param'][name] = tensor.name
      # op_des['param'][name] = tensor.description()

    op_des['attrs'] = {}
    for name, attr in self._attrs.items():
      value = serialize_attr(attr.value)
      op_des['attrs'][name.value] = value

    return op_des

  def set_optype(self, value):
    self._type = value

  # def set_attr(self, attr_name: Enum, attr_value: List[Any]) -> None:
  #   try:
  #     self._attrs[attr_name].value = attr_value
  #   except Exception as e:
  #     raise RuntimeError(
  #         f"failed to set attr '{attr_name.value}' of {self._type}: {str(e)}")

  # def get_attr(self, attr_name: Enum) -> Any:
  #   try:
  #     ret = self._attrs[attr_name].value
  #   except Exception as e:
  #     raise RuntimeError(
  #         f"failed to get attr '{attr_name.value}' of {self._type}: {str(e)}")
  #   else:
  #     return ret

  def _check_attr_valid(self,
                        attr_name: str,
                        attr_value: Optional[List[Any]] = None) -> None:
    if attr_name not in self._attrs:
      raise KeyError(
          f"the attr '{attr_name}' not in op '{self.__class__.__name__}'")
    # if attr_value and not isinstance(attr_value, list):
    #   raise TypeError(f"the attr value of {attr_name} should be list")

  def has_attr(self, attr_name: Union[Enum, str]):
    if isinstance(attr_name, Enum):
      return attr_name in self._attrs
    elif isinstance(attr_name, str):
      if hasattr(self, 'AttrName'):
        for name in getattr(self, 'AttrName'):
          if attr_name == name.value:
            return True
      return False
    else:
      raise ValueError('"attr_name" must be either Enum or string')

  def get_attr(self, attr_name) -> Any:
    self._check_attr_valid(attr_name)
    return self._attrs[attr_name].value

  def set_attr(self, attr_name: str, attr_value: List[Any]) -> None:
    self._check_attr_valid(attr_name)
    self._set_attr_user(attr_name, attr_value)
    self._attrs[attr_name].value = attr_value

  def update_attr(self, attr_name, attr_value):
    self._check_attr_valid(attr_name)
    self._release_attr_user(attr_name)
    self._set_attr_user(attr_name, attr_value)
    self._attrs[attr_name].value = attr_value

  def _release_attr_user(self, attr_name):
    def _release(attr_value):
      if isinstance(attr_value, (list, tuple)):
        for val in attr_value:
          _release(val)

      if not isinstance(attr_value, Tensor):
        return

      remove_uses = []
      for attr_use in attr_value.attr_uses:
        if attr_use.user == self and attr_use.attr_name == attr_name:
          remove_uses.append(attr_use)

      for use in remove_uses:
        attr_value.attr_uses.remove(use)

    if isinstance(attr_name, str):
      attr_value = self.get_config(attr_name)
    else:
      attr_value = self.get_attr(attr_name)

    _release(attr_value)

  def _set_attr_user(self, attr_name, attr_value):
    if isinstance(attr_value, (list, tuple)):
      for val in attr_value:
        self._set_attr_user(attr_name, val)

    if not isinstance(attr_value, Tensor):
      return

    for attr_use in attr_value.attr_uses:
      if attr_use.user == self and attr_use.attr_name == attr_name:
        return
    attr_value.attr_uses.append(AttrUser(self, attr_name))

  def get_config(self, config_name: str) -> Any:
    return getattr(self, config_name)

  def set_config(self, config_name: str, value: Any) -> None:
    if config_name not in self._configs:
      self._configs.append(config_name)
      setattr(self, config_name, value)
      self._set_attr_user(config_name, value)
    else:
      self._release_attr_user(config_name)
      self._set_attr_user(config_name, value)
      setattr(self, config_name, value)

  def get_param(self, name: Enum):
    return self._params[name]

  def set_param(self, name: Enum, value: Union[Tensor, List[Tensor]]):
    # String param name is allowed, this because custom LSTM may use any
    # weight names, so we can't pre-define ParamName before we see them.
    # In this case, we use the original weight name as key to save params.
    if not isinstance(name, (str, Enum)):
      raise ValueError("Invalid param name: {}".format(type(name)))
    #if not isinstance(value, Tensor):
    #  raise ValueError("Param must be 'Tensor', but given {}".format(
    #      type(value)))
    self._params[name] = value

  def set_param_from_data(self,
                          name: Enum,
                          data: np.ndarray,
                          framework_name: Optional[str] = None):

    if name in self._params:
      self._params[name].from_ndarray(data)
    else:
      new_tensor = Tensor()
      new_tensor.from_ndarray(data)
      if framework_name:
        new_tensor.name = framework_name
      else:
        new_tensor.name = self._params[name].name
      self._params[name] = new_tensor

  def set_param_from_des(self,
                         param_name: Enum,
                         description: Dict[str, Any],
                         fw_name: Optional[str] = None):
    if not isinstance(description, dict):
      raise TypeError("'description' must be a dictionary")

    new_tensor = Tensor()
    new_tensor.from_des(**description)
    if fw_name:
      new_tensor.name = fw_name
    else:
      new_tensor.name = self._params[param_name].name
    self._params[param_name] = new_tensor

  def has_native_params(self):
    return hasattr(self, 'ParamName')

  def is_xir_attr(self, attr_name: Enum):
    return self._attrs[attr_name].is_xir_attr

  @property
  def type(self):
    return self._type

  @type.setter
  def type(self, value):
    self._type = value

  @property
  def attrs(self):
    return self._attrs

  @property
  def params(self):
    return self._params

  @property
  def configs(self):
    return self._configs

  @property
  def is_custom_op(self):
    return self._is_custom_op

  @is_custom_op.setter
  def is_custom_op(self, is_custom_op):
    self._is_custom_op = is_custom_op
