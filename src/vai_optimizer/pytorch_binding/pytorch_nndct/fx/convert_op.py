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
import numbers
from collections import namedtuple
import torch
from nndct_shared.base import NNDCT_OP
from nndct_shared.nndct_graph import Tensor
from pytorch_nndct.parse.torch_op_def import *
from pytorch_nndct.fx.translator_utils import convert_dtype, convert_shape


def int2tuple(obj, tuple_size):
  if isinstance(obj, numbers.Integral):
    return (obj, ) * tuple_size
  else:
    return obj


def convert_real_tensor(name, real_tensor):
  if real_tensor is None:
    return  
  tensor = Tensor(
    name=name, 
    shape=convert_shape(real_tensor.shape),
    dtype=convert_dtype(real_tensor.dtype),
    data = real_tensor.cpu().detach().numpy(),
    # device=get_tensor_meta_info(input_meta, "device"),
    requires_grad=real_tensor.requires_grad)
  return tensor


_OP_CONVERTER_DICT = {}
OpConvertInfo = namedtuple("OpConvertFnInfo", ["convert_fn", "args_kwargs_mapping_fn"])


def create_op(gm, op, target, args, kwargs):
  def get_new_target():
    if op == "call_module":
      return gm.get_submodule(target).__class__
    elif op == "call_method":
      return getattr(torch.Tensor, target)
    else:
      return target

  convert_info = _OP_CONVERTER_DICT.get((op, get_new_target()), _OP_CONVERTER_DICT[(op, "")])
  normalized_kwargs = kwargs
  if convert_info.args_kwargs_mapping_fn:
    if op == "call_module":
      mod = gm.get_submodule(target)
      normalized_kwargs = convert_info.args_kwargs_mapping_fn(target, mod, args, kwargs)

  return convert_info.convert_fn(*args, **normalized_kwargs)


def register_nndct_op(op, target, args_kwargs_mapping_fn=None):
  def register(op_convert_func):
    op_and_target = (op, target)
    assert op_and_target not in _OP_CONVERTER_DICT.keys()
    _OP_CONVERTER_DICT[op_and_target] = OpConvertInfo(op_convert_func, args_kwargs_mapping_fn)
    return op_convert_func
  return register


@register_nndct_op(op="placeholder", target="")
def input_arg(*args, **kwargs):
  op = TorchBaseOperation(NNDCT_OP.INPUT)
  return op


@register_nndct_op(op="call_method", target="")
def call_method(*args, **kwargs):
  op = TorchBaseOperation(NNDCT_OP.CALL_METHOD)
  return op

@register_nndct_op(op="call_module", target="")
def call_module(*args, **kwargs):
  op = TorchBaseOperation(NNDCT_OP.CALL_MODULE)
  return op


@register_nndct_op(op="call_function", target="")
def call_function(*args, **kwargs):
  op = TorchBaseOperation(NNDCT_OP.CALL_FUNCTION)
  return op

@register_nndct_op(op="output", target="")
def output(*args, **kwargs):
  op = TorchBaseOperation(NNDCT_OP.RETURN)
  return op

def conv_nd_mapping_fn(mod_qual_name, mod, args, kwargs):

  weight_tensor = convert_real_tensor(".".join([mod_qual_name, "weight"]), mod.weight)
  bias_tensor = convert_real_tensor(".".join([mod_qual_name, "bias"]), mod.bias)
 
  nd = weight_tensor.ndim - 2
  stride = int2tuple(mod.stride, nd)
  padding = int2tuple(mod.padding, nd)
  dilation = int2tuple(mod.dilation, nd)
  groups = mod.groups
  return {"input": kwargs["input"], 
          "weight": weight_tensor,
          "bias": bias_tensor,
          "stride": stride,
          "padding": padding,
          "dilation": dilation,
          "groups": groups
          }


@register_nndct_op(op="call_module", target=torch.nn.Conv2d, args_kwargs_mapping_fn=conv_nd_mapping_fn)
def conv2d(*, input, weight, bias, stride, padding, dilation, groups):
   return _convolution(input, weight, bias, stride, padding, dilation, False, None, groups)


def _convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
  
  weight_size = weight.shape
  if transposed:
    weight_size[0], weight_size[1] = weight_size[1], weight_size[0]
    if weight_size[0] == 1 and groups == weight_size[1]:
        if weight.ndim == 4:
          op = TorchConvTranspose2d(NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D)
        elif weight.ndim == 5:
          op = TorchConvTranspose3d(NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D)
        elif weight.ndim == 3:
          raise NotImplementedError("Depthwise_ConvTranpose1D is unsupported")
    else:
      if weight.ndim == 4:
        op = TorchConvTranspose2d(NNDCT_OP.CONVTRANSPOSE2D)
      elif weight.ndim == 5:
        op = TorchConvTranspose3d(NNDCT_OP.CONVTRANSPOSE3D)
      elif weight.ndim == 3:
        raise NotImplementedError("ConvTranpose1D is unsupported")

    op.set_config("output_padding", list(output_padding))
    op.set_config('in_channels', weight_size[1])
    op.set_config('out_channels', weight_size[0] * groups)
  else:
    if weight_size[1] == 1 and groups == weight_size[0]:
      if weight.ndim == 4:
        op = TorchConv2d(NNDCT_OP.DEPTHWISE_CONV2D)
      elif weight.ndim == 5:
        op = TorchConv3d(NNDCT_OP.DEPTHWISE_CONV3D)
      elif weight.ndim == 3:
        op = TorchConv1d(NNDCT_OP.DEPTHWISE_CONV1D)
    else:
      if weight.ndim == 4:
        op = TorchConv2d(NNDCT_OP.CONV2D)
      elif weight.ndim == 5:
        op = TorchConv3d(NNDCT_OP.CONV3D)
      elif weight.ndim == 3:
        op = TorchConv1d(NNDCT_OP.CONV1D)
        
    op.set_config('in_channels', weight_size[1] * groups)
    op.set_config('out_channels', weight_size[0])

    

  # Should add weight first
  op.set_param(op.ParamName.WEIGHTS, weight)
  if bias is not None:
    op.set_config('bias', True)
    op.set_param(op.ParamName.BIAS, bias)
  else:
    op.set_config('bias', False)

  op.set_config('dilation', list(dilation))
  op.set_config('kernel_size', list(weight_size[2:]))
  op.set_config('stride', list(stride))
  
  op.set_config('groups', groups)
  op.set_config('padding', list(padding))

  return op

@register_nndct_op(op="call_function", target=torch.nn.functional.relu)
def relu(*, input, inplace):
  # s
  op = TorchReLU()
  op.set_config('inplace', inplace)
  return op


def linear_mapping_fn(mod_qual_name, mod, args, kwargs):
  weight_tensor = convert_real_tensor(".".join([mod_qual_name, "weight"]), mod.weight)
  bias_tensor = convert_real_tensor(".".join([mod_qual_name, "bias"]), mod.bias)
  return {"input": kwargs["input"], 
          "weight": weight_tensor,
          "bias": bias_tensor
          }

@register_nndct_op(op="call_module", target=torch.nn.Linear, args_kwargs_mapping_fn=linear_mapping_fn)
def linear(*, input, weight, bias):
  op = TorchLinear()
  weight_size = weight.shape
  op.set_param(op.ParamName.WEIGHTS, weight)
  if bias is None:
    op.set_config("bias", False)
  else:
    op.set_config("bias", True)
    op.set_param(op.ParamName.BIAS, bias)

  op.set_config('out_features', weight_size[0])
  op.set_config('in_features', weight_size[1])
  return op

@register_nndct_op(op="call_function", target=torch.flatten)
def flatten(*, input, start_dim, end_dim):
  op = TorchFlatten()
  op.set_config('input', input)
  op.set_config("start_dim", start_dim)
  op.set_config("end_dim", end_dim)
  return op


@register_nndct_op(op="call_function", target=torch.nn.functional.adaptive_avg_pool2d)
def adaptive_avg_pool2d(*, input, output_size):
    op = TorchAdaptiveAvgPool()
    op.set_config("output_size", output_size)
    return op


@register_nndct_op(op="call_function", target=torch.nn.functional.max_pool2d)
def maxpool2d(*, input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
  op = TorchMaxPool()
  op.set_config("return_indices", return_indices)
  op.set_config('kernel_size', list(int2tuple(kernel_size, 2)))
  if not stride:
    op.set_config('stride', list(int2tuple(kernel_size, 2)))
  else:
    op.set_config('stride', list(int2tuple(stride, 2)))

  if ceil_mode:
    op.set_config('ceil_mode', True)
  else:
    op.set_config('ceil_mode', False)

  op.set_config('padding', list(int2tuple(padding, 2)))
  op.set_config('dilation', list(int2tuple(dilation, 2)))

  return op

@register_nndct_op(op="call_function", target=torch.add)
def add(*, input, other, alpha):
  op = TorchAdd()
  op.set_config('input', input)
  op.set_config('other', other)
  op.set_config('alpha', alpha)
  return op




  
