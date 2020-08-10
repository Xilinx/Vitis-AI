import json
import numpy as np

from collections import OrderedDict
from enum import Enum, auto
from functools import partial
from typing import Dict, List, Callable, Optional, Union, Any, Set

from nndct_shared.nndct_graph.base_tensor import Tensor


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
    if len(value_mem) == 1:
      return value_mem[0]
    else:
      return value_mem[:]


class AutoName(Enum):

  def _generate_next_value_(name, start, count, last_values):
    return name.lower()


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
               read_and_write_value_func: Optional[Callable] = None) -> None:
    self._name = name
    self._type = value_type
    self._size = size
    self._value_mem = value_mem
    self._occurence_type = occurence_type
    self._annotation = annotation

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
  
  @property
  def type(self):
    return self._type
  
  @property
  def value(self) -> Any:
    return self._read_and_write_value_func(in_out=1)

  @value.setter
  def value(self, value):
    if not isinstance(value, (list, tuple, set)):
      value = [value]
    else:
      value = list(value)
    if not isinstance(value[0], self._type):
      raise TypeError(
          f"The type of attr '{self._name.value}' should be {self._type} instead of {type(value[0])}"
      )
    if self._size is not None and len(value) != self._size:
      raise ValueError(
          f"the length of  value of {self._name.value} is not equal to {self._size}"
      )
    self._read_and_write_value_func(in_out=0, attr_value=value)

class Operation(object):
  """Object that takes one or more tensors as input and outputs tensors."""

  def __init__(self, optype: str):
    self._attrs: Dict[Enum, NndctIrAttr] = {}
    # Attr names defined in the framework.
    self._configs: Set[str] = set()
    self._params: Dict[Enum, Tensor] = OrderedDict()
    self._type = optype

    self._export_attr_and_param()

  def _export_attr_and_param(self):
    """Wrappers for self._attrs and self._params."""

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

    # Get/set attr/param by indexing.
    # e.g. op.attr['out_dim'] = 32
    #      op.param['weight']
    if hasattr(self, 'AttrName'):
      self.attr = AttrParamIndexer(
          self.get_attr, self.set_attr, self.AttrName)

    if hasattr(self, 'ParamName'):
      self.param = AttrParamIndexer(
          self.get_param, self.set_param, self.ParamName)

  def __repr__(self):
    return json.dumps(self.description(), indent=4, separators=(',', ': '))

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

    op_des['configs'] = {}
    for name in self._configs:
      value = serialize_attr(self.get_config(name))
      op_des['configs'][name] = value

    return op_des

  def set_optype(self, value):
    self._type = value

  def set_attr(self, attr_name: Enum, attr_value: List[Any]) -> None:
    try:
      self._attrs[attr_name].value = attr_value
    except Exception as e:
      raise RuntimeError(
          f"failed to set attr '{attr_name.value}' of {self._type}: {str(e)}")

  def get_attr(self, attr_name: Enum) -> Any:
    try:
      ret = self._attrs[attr_name].value
    except Exception as e:
      raise RuntimeError(
          f"failed to get attr '{attr_name.value}' of {self._type}: {str(e)}")
    else:
      return ret

  def _check_attr_valid(self,
                        attr_name: str,
                        attr_value: Optional[List[Any]] = None) -> None:
    if attr_name not in self._attrs:
      raise KeyError(
          f"the attr '{attr_name}' not in op '{self.__class__.__name__}'")
    # if attr_value and not isinstance(attr_value, list):
    #   raise TypeError(f"the attr value of {attr_name} should be list")

  def get_attr(self, attr_name) -> Any:
    self._check_attr_valid(attr_name)
    return self._attrs[attr_name].value

  def set_attr(self, attr_name: str, attr_value: List[Any]) -> None:
    self._check_attr_valid(attr_name, attr_value)
    self._attrs[attr_name].value = attr_value

  def get_config(self, config_name: str) -> Any:
    return getattr(self, config_name)

  def set_config(self, config_name: str, value: Any) -> None:
    self._configs.add(config_name)
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
