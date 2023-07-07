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

import numpy as np
import weakref
from enum import Enum, unique, auto
import copy

class Tensor(object):
  """A wrapper of np.ndarray used in two ways:
  - The outputs of an operation.
  - The parameters of an operation.
  In the former case, you can use `tensor.node` to get the node that
  outputs this tensor.
  In the latter case, `tensor.node` is None.
  For getting raw ndarray, call `tensor.data`.
  """
  

  def __init__(self,
               name=None,
               shape=None,
               dtype=None,
               device=None,
               requires_grad=None,
               data=None,
               node=None,
               layout=None):
    self._node = weakref.ref(node) if node else node
    self._name = name
    self._shape = shape
    self._data = data
    self._dtype_map = {
        np.dtype('float64'): 'float64',
        np.dtype('float32'): 'float32',
        np.dtype('float16'): 'float16',
        np.dtype('complex64'): 'complex64',
        np.dtype('int64'): 'int64',
        np.dtype('int32'): 'int32',
        np.dtype('int16'): 'int16',
        np.dtype('int8'): 'int8',
        np.dtype('bool'): 'bool'
    }
    if dtype in self._dtype_map:
      self._dtype = self._dtype_map[dtype]
    else:
      self._dtype = dtype

    self._device = device
    self._requires_grad = requires_grad
    self._offset = 0
    self._uses = []
    self._attr_uses = []
    self._device = device

  def __deepcopy__(self, memo):
    raise NotImplementedError("Deep copy is prohibited, use `clone_from` instead.")  
  
  def clone_from(self, src_tensor):
    self._shape = src_tensor._shape
    self._data = copy.deepcopy(src_tensor._data)
    self._dtype = src_tensor._dtype
    self._device = src_tensor._device
    self._requires_grad  = src_tensor._requires_grad
    
  
  def from_ndarray(self, data):
    if not isinstance(data, np.ndarray):
      raise TypeError("'data' must be a numpy ndarray")
    self._data = np.copy(data)
    self._dtype = self._dtype_map[self._data.dtype]
    self._shape = list(self._data.shape)

  def from_tensor(self, tensor):
    self._dtype = tensor.dtype
    self._shape = tensor.shape

  def from_des(self, shape, dtype):
    self._shape = shape
    self._dtype = dtype

  def transpose(self, axes=None):
    trans_data = None
    if self._data is not None:
      trans_data = self._data.transpose(axes)
      trans_data = np.ascontiguousarray(trans_data)
      trans_shape = list(trans_data.shape)
    else:
      trans_shape = [self._shape[i] for i in axes]
    self._data = trans_data
    self._shape = trans_shape

  def clean_data(self):
    self._data = None

  def __str__(self):
    return "Tensor: {}(shape={}, dtype={})".format(
        self._name if self._name else "", self._shape, self._dtype)

  def description(self):
    desp = {}
    desp['name'] = self._name
    desp['shape'] = self._shape
    desp['dtype'] = self._dtype
    desp['node'] = self.node.name if self.node else None
    return desp

  
  def is_real_tensor(self):
    return self.is_complete_tensor() or self.dtype == "tensor"

  def is_list_type(self):
    return "list" in self.dtype
  
  def is_complete_tensor(self) -> bool:
    # not necessary to hold real data for completeTensor
    return True if self.shape and self.dtype else False

  def is_param_tensor(self) -> bool:
    return True if self._node is None else False

  @property
  def shape(self):
    return self._shape

  @shape.setter
  def shape(self, shape):
    self._shape = shape

  @property
  def ndim(self):
    return len(self._shape) if self._shape else None

  @property
  def dtype(self):
    return self._dtype

  @dtype.setter
  def dtype(self, dtype):
    self._dtype = dtype

  @property
  def data(self):
    return self._data

  @data.setter
  def data(self, value):
    if isinstance(value, np.ndarray):
      self.from_ndarray(value)
    elif isinstance(value, Tensor):
      self.from_tensor(value)
    elif isinstance(value, (int, float, bool)):
        #raise ValueError(f"Accept [int, float, bool] type data, but {type(value)} is given")
      self._data = value
      self._shape = []

  @property
  def node(self):
    return self._node() if self._node is not None else None

  @node.setter
  def node(self, value):
    self._node = weakref.ref(value) if value else value

  @property
  def name(self):
    return self._name

  @name.setter
  def name(self, name):
    self._name = name

  @property
  def device(self):
    return self._device

  @device.setter
  def device(self, device):
    self._device = device

  @property
  def requires_grad(self):
    return self._requires_grad

  @requires_grad.setter
  def requires_grad(self, need_grad):
    self._requires_grad = need_grad


  @property
  def offset(self):
    return self._offset
  
  
  @offset.setter
  def offset(self, offset):
    self._offset = offset
    
  @property
  def uses(self):
    return self._uses
  
  @property
  def owning_graph(self):
    return self.node.owning_graph
  
  @property
  def attr_uses(self):
    return self._attr_uses

  def replace_first_use_with(self, new_tensor):
    assert self.owning_graph is new_tensor.owning_graph
    u = self.uses[0]
    u.user.in_tensors[u.offset] = new_tensor
    new_tensor.uses.append(u)
    self.uses.pop(0) 
  
  def replace_uses_with(self, new_tensor):
    assert self is not new_tensor
    while len(self.uses) > 0:
      self.replace_first_use_with(new_tensor)
   
    self.replace_attr_uses_with(new_tensor)

  def replace_attr_with_new_tensor_v2(self, attr_use, new_tensor):
    def _replace(attr_name, attr_value):
      if isinstance(attr_value, list):
        new_attr_value = []
        for value in attr_value:
          if self is value:
            new_value = _replace(attr_name, value)
            new_attr_value.append(new_value)
          elif isinstance(value, (tuple, list)):
            new_value = _replace(attr_name, value)
            new_attr_value.append(new_value)
          else:
            new_attr_value.append(value)
        return new_attr_value

      elif isinstance(attr_value, tuple):
        new_attr_value = []
        for value in list(attr_value):
          if self is value:
            new_value = _replace(attr_name, value)
            new_attr_value.append(new_value)
          elif isinstance(value, (tuple, list)):
            new_value = _replace(attr_name, value)
            new_attr_value.append(new_value)
          else:
            new_attr_value.append(value)
        new_attr_value = tuple(new_attr_value)
        return new_attr_value
      else:
        if self is not attr_value:
          if attr_name == attr_use.attr_name:
            self.attr_uses.remove(attr_use)
            return attr_value
        else:
          if attr_use not in new_tensor.attr_uses:
            new_tensor.attr_uses.append(attr_use)
            self.attr_uses.remove(attr_use)
          return new_tensor


    if isinstance(attr_use.attr_name, str):
      attr_value = attr_use.user.get_config(attr_use.attr_name)
    else:
      attr_value = attr_use.user.get_attr(attr_use.attr_name)

    new_attr_value = _replace(attr_use.attr_name, attr_value)
    
    if isinstance(attr_use.attr_name, str):
      attr_use.user.set_config(attr_use.attr_name, new_attr_value)
    else:
      attr_use.user.update_attr(attr_use.attr_name, new_attr_value)



  def replace_attr_with_new_tensor(self, attr_use, new_tensor):
    def _replace(attr_name, attr_value):
      if isinstance(attr_value, list):
        if self in attr_value:
          for i in range(len(attr_value)):
            if attr_value[i] is self:
              attr_value[i] = new_tensor
              if attr_use not in new_tensor.attr_uses:
                new_tensor.attr_uses.append(attr_use)
                self.attr_uses.remove(attr_use)
          return
        else:
          for val in attr_value:
            _replace(attr_name, val)
            
      if self is not attr_value:
        if attr_name == attr_use.attr_name:
          self.attr_uses.remove(attr_use)
        return
      
      if isinstance(attr_name, str):
        attr_use.user.set_config(attr_name, new_tensor)
      else:
        attr_use.user.update_attr(attr_name, new_tensor)
    
    if isinstance(attr_use.attr_name, str):
      attr_value = attr_use.user.get_config(attr_use.attr_name)
    else:
      attr_value = attr_use.user.get_attr(attr_use.attr_name)
    

    _replace(attr_use.attr_name, attr_value)   
  

  def replace_first_attr_use_with(self, new_tensor):
    attr_use = self.attr_uses[0]
    self.replace_attr_with_new_tensor_v2(attr_use, new_tensor)
  
  
  def replace_attr_uses_with(self, new_tensor):
    while len(self.attr_uses) > 0:
      self.replace_first_attr_use_with(new_tensor)
