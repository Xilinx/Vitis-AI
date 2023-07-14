
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
import os
import sys as _sys
from typing import TypeVar, Optional
from .exception import DefineOptionError

T = TypeVar('T')
_OPTION_PREFFIX = "nndct_"

class Option(object):
  """NNDCT option definition.
  
  Attribute:
  
    name(str): option name
    dtype(str, int, float, bool): option type 
    default(T): default value of option
    action(str): 'store_true' / 'store_false' only work when dtype is 'bool' [default=None]
    help(str): description of option  [default=None]
    framework(str): 'torch' / 'tensorflow' / 'all' [default='all'] 

  Raises:

    DefineOptionError

"""
  def __init__(self, name: str, dtype: type, default: T, action: Optional[str] = None, framework: str = "all", help: Optional[str] = None, env=None):
    self._name = _OPTION_PREFFIX + name
    self._dtype = dtype
    self._default = default
    self._action = action
    self._framework = framework
    self._help = help
    self._env = env
    try:
      self._check_attribute_validataion_()
    except DefineOptionError as e:
      print(e)
      _sys.exit(1)
  
  def __str__(self):
    return f"""--{self._name} : {self._help} (default={self._default})"""
                  
  
  def _check_attribute_validataion_(self):
    if self._dtype not in [str, int, float, bool]:
      raise DefineOptionError(self._name, msg=r"The dtype should be 'int/float/bool/string'.")
    
    if self._action not in [None, "store_true", "store_false"]:
      raise DefineOptionError(self._name, msg=r"The action value should be ''store_true' / 'store_false''.")

    if self._framework not in ["tensorflow", "torch", "all"]:
      raise DefineOptionError(self._name, msg=r"The framewok should be ''tensorflow''/''torch''/''all''.")

    
    if type(self._default) != self._dtype:
      raise DefineOptionError(self._name, msg=r"The default value type should be the same with dtype.")
    
    if self._dtype != bool and self._action is not None:
      raise DefineOptionError(self._name, msg=r"The action is only valid for bool type option.")
  

  def get_env_value(self):
    if self._dtype == str:
      return os.getenv(self._env, default=self._default)
    elif self._dtype in [int, float]:
      data = os.getenv(self._env, default=self._default)
      return self._dtype(data)
    elif self._dtype == bool:
      data = os.getenv(self._env)
      if data is None:
        return self._default
      else:
        return {"true": True,
                "false": False,
                "0": False}.get(data.lower(), True)
  @property
  def dtype(self):
    return self._dtype
  
  @property
  def action(self):
    return self._action
  
  @property
  def framework(self):
    return self._framework
  
  @property
  def value(self):
    if hasattr(self, '_value'):
      return self._value
    elif self._env is not None:
      return self.get_env_value()
    else:
      return self._default
  
   
    
  @value.setter
  def value(self, value):
    if value is None:
      self._value = True if self._action == "store_true" else False
    else:
      self._value = value  
    
