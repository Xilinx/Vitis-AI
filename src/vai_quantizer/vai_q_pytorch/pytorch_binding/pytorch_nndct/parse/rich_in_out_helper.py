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

import re
import torch
import collections
from typing import List, Dict
from dataclasses import dataclass


class StandardInputData(object):
  def __init__(self, in_args, in_kwargs, device=None) -> None:
    '''
    example1:
      model(a)
        args is (a,)
        kwargs is {}

    example2:
      model(a, b, c)
        args is (a, b, c)
        kwargs is {}

    example3:
      model(a, d=0, f=None)
        args is (a,)
        kwargs is {'d':0,'f':None}

    example4:
      model(a, b, c, d=0, f=None)
        args is (a,b,c)
        kwargs is {'d':0,'f':None}
    
    example5:
      model(d=0, f=None)
        args is ()
        kwargs is {'d':0,'f':None}
    '''
    super().__init__()
    args = in_args
    kwargs = in_kwargs
    if args is None: args = ()
    if kwargs is None: kwargs = {}
    if not isinstance(args, tuple):
      args = (args,)
    assert(isinstance(kwargs,dict))

    if device is None:
      args_device = StandardInputData.get_data_device(args)
      kwargs_device = StandardInputData.get_data_device(kwargs)
      if args_device is None and kwargs_device is None:
        device = torch.device('cpu')
      elif args_device is not None:
        device = args_device
      else:
        device = kwargs_device
    
    if not torch.cuda.is_available():
        device = torch.device("cpu")

    self._args = StandardInputData.to_device(StandardInputData.checkType(args), device)
    self._kwargs = StandardInputData.to_device(StandardInputData.checkType(kwargs), device)

  @classmethod
  def get_data_device(cls, data):
    device = None
    if isinstance(data,(tuple,list)):
      for v in data:
        device = StandardInputData.get_data_device(v)
        if device is not None:
          return device
    if isinstance(data,(dict)):
      for k in data:
        device = StandardInputData.get_data_device(data[k])
        if device is not None:
          return device
    if isinstance(data,torch.Tensor):
      device = data.device
    if isinstance(data,StandardInputData):
      device = StandardInputData.get_data_device(data.data)
    return device

  @classmethod
  def checkTraceType(cls, value):
    if not isinstance(value, (tuple, list, torch.Tensor)):
      raise TypeError(f"Type of input_args should be tuple/list/torch.Tensor.")
  
  @classmethod
  def checkType(cls, value):
    if value is None:
      return
    if isinstance(value, (Dict)):
      # check none ,and remove
      for k in value:
        StandardInputData.checkType(value[k])
    else:
      StandardInputData.checkTraceType(value)
    return value

  @classmethod
  def to_device(cls, value, device):
    if isinstance(value, (Dict)):
      for k in value:
        value[k] = StandardInputData.to_device(value[k], device)
    elif isinstance(value, (bool)):
        value = torch.BoolTensor([value]).to(device)
    else:
      if value is not None:
        if isinstance(value, torch.Tensor):
          value = value.to(device)
        else:
          is_tuple = True if isinstance(value, tuple) else False
          value = list(value)
          for i in range(len(value)):
            inp = StandardInputData.to_device(value[i], device)
            value[i] = inp
          if is_tuple:
            value = tuple(value)
    return value

  @property
  def args(self):
    return self._args

  @property
  def kwargs(self):
    return  self._kwargs

  @property
  def data(self):
    return {
      'args': self.args,
      'kwargs': self.kwargs
    }
  
  def make_flatten_data(self):
    data, schema = flatten_to_tuple(self.data)
    return data, schema


class FlattenInOutModelForTrace(torch.nn.Module):
    @classmethod
    def getModelName(cls):
      return 'FlattenInOutModelForTrace'

    @classmethod
    def getOriginModelNameFormString(cls, data_str):
      names = re.findall(r"nndct_st_([\w_]+)_ed", data_str)
      if len(names) > 0:
        return names[0]
      else:
        return None

    @classmethod
    def check_need_recovery_name(cls, name):
      return 'nndct_st_' in name or 'FlattenInOutModelForTrace' in name

    @classmethod
    def recovery_tensor_name(cls, name):
      return re.sub(r"nndct_st_[\w_]+_ed\.", '', name)

    @classmethod
    def recovery_node_scope_name(cls, scope_name):
      real_class_name = re.findall(r"FlattenInOutModelForTrace/(.*)?\[nndct_st_[\w_]+_ed\]", scope_name)
      if len(real_class_name) > 0:
        scope_name = re.sub(r"FlattenInOutModelForTrace/(.*)?\[nndct_st_[\w_]+_ed\]", real_class_name[0], scope_name)

      real_model_name = re.findall(r".*FlattenInOutModelForTrace::/(.*::)nndct_st_[\w_]+_ed", scope_name)
      if len(real_model_name) > 0:
        scope_name = re.sub(r".*FlattenInOutModelForTrace::/(.*::)nndct_st_[\w_]+_ed", real_model_name[0], scope_name)
      return scope_name

    def __init__(self, inner_model, input_schema) -> None:
        super().__init__()
        self.module_name = "nndct_st_" + inner_model._get_name()  + "_ed" 
        setattr(self, self.module_name, inner_model)
        self.input_schema = input_schema
        self.training = inner_model.training


    def forward(self, *flatten_input):
        input = self.input_schema(flatten_input)
        output = getattr(self, self.module_name)(*input['args'], **input['kwargs'])  
        flatten_output, _ = flatten_to_tuple(output)
        return flatten_output   

class RecoveryModel(torch.nn.Module):
    def __init__(self, inner_model, output_schema, device) -> None:
        super().__init__()
        self.inner_model = inner_model
        self.output_schema = output_schema
        self.device = device
        self.training = inner_model.training
    
    def forward(self, *args, **kwargs):
        input_data = StandardInputData(args, kwargs, self.device)
        flatten_input, _ = input_data.make_flatten_data()
        flatten_output = self.inner_model(*flatten_input)
        if torch.is_tensor(flatten_output):
            return self.output_schema((flatten_output,))
        else:
            return self.output_schema(flatten_output)
    
    def __getattr__(self, name):
      if name in self.__dict__.keys():
        return self.__dict__[name]
      if name in self._modules.keys():
        return self._modules[name]
      return getattr(self.inner_model,name)


@dataclass
class Schema:

    @classmethod
    def flatten(cls, obj):
        raise NotImplementedError

    def __call__(self, values):
        raise NotImplementedError

    @staticmethod
    def _concat(values):
        ret = ()
        sizes = []
        for v in values:
            assert isinstance(v, tuple), "Flattened results must be a tuple"
            ret = ret + v
            sizes.append(len(v))
        return ret, sizes

    @staticmethod
    def _split(values, sizes):
        if len(sizes):
            expected_len = sum(sizes)
            assert (
                len(values) == expected_len
            ), f"Values has length {len(values)} but expect length {expected_len}."
        ret = []
        for k in range(len(sizes)):
            begin, end = sum(sizes[:k]), sum(sizes[: k + 1])
            ret.append(values[begin:end])
        return ret


@dataclass
class HandleListType(Schema):
    schemas: List[Schema]
    sizes: List[int]

    def __call__(self, values):
        values = self._split(values, self.sizes)
        if len(values) != len(self.schemas):
            raise ValueError(
                f"Values has length {len(values)} but schemas " f"has length {len(self.schemas)}!"
            )
        values = [m(v) for m, v in zip(self.schemas, values)]
        return list(values)

    @classmethod
    def flatten(cls, obj):
        res = [flatten_to_tuple(k) for k in obj]
        values, sizes = cls._concat([k[0] for k in res])
        return values, cls([k[1] for k in res], sizes)


@dataclass
class HandleTupleType(HandleListType):
    def __call__(self, values):
        return tuple(super().__call__(values))


@dataclass
class IdentitySchema(Schema):
    def __call__(self, values):
        try:
          return values[0]
        except:
          return values

    @classmethod
    def flatten(cls, obj):
        return (obj,), cls()


@dataclass
class HandleDictType(HandleListType):
    keys: List[str]

    def __call__(self, values):
        values = super().__call__(values)
        return dict(zip(self.keys, values))

    @classmethod
    def flatten(cls, obj):
        for k in obj.keys():
            if not isinstance(k, str):
                raise KeyError("Only support flattening dictionaries if keys are str.")
        keys = sorted(obj.keys())
        values = [obj[k] for k in keys]
        ret, schema = HandleListType.flatten(values)
        return ret, cls(schema.schemas, schema.sizes, keys)

@dataclass
class HandleOrderDictType(HandleListType):
    keys: List[str]

    def __call__(self, values):
        values = super().__call__(values)
        return dict(zip(self.keys, values))

    @classmethod
    def flatten(cls, obj):
        for k in obj.keys():
            if not isinstance(k, str):
                raise KeyError("Only support flattening dictionaries if keys are str.")
        keys = obj.keys()
        values = [obj[k] for k in keys]
        ret, schema = HandleListType.flatten(values)
        return ret, cls(schema.schemas, schema.sizes, keys)

def flatten_to_tuple(data):
    support_list = [
        ((str, bytes), IdentitySchema),
        (list, HandleListType),
        (tuple, HandleTupleType),
        (collections.OrderedDict, HandleOrderDictType),
        (collections.abc.Mapping, HandleDictType),
    ]
    for support_type, schema in support_list:
        if isinstance(data, support_type):
            handle = schema
            break
    else:
        handle = IdentitySchema

    return handle.flatten(data)
