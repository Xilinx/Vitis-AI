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

import math
import sys
import torch
import numpy as np
from typing import Callable, Dict
from nndct_shared.nndct_graph import Node, Tensor, Operation
from nndct_shared.base import GLOBAL_MAP, NNDCT_KEYS
from nndct_shared.utils import tensor_util
from .torch_op_def import *
from .parse_utils import *
from pytorch_nndct.utils import SchemaHelper, TorchOpClassType, convert_type_str



def convert_dtype(dtype):
  r"""convert torch dtype to nndct dtype"""
  return {
      'torch.float': 'float32',
      'torch.double': 'float64',
      'torch.int': 'int32',
      'torch.long': 'int64'
  }.get(dtype, dtype)


class TensorConvertor(object):
  r"""convert value to nndct_tensor"""

  def __call__(self, scope, value):
    if isinstance(value.data, torch.Tensor):
      nndct_tensor = Tensor(
          name=get_full_name(scope, value.name),
          shape=value.shape,
          dtype=value.dtype,
          data=value.data.cpu().numpy(),
          layout=value.layout)
    else:
      nndct_tensor = Tensor(
          name=get_full_name(scope, value.name),
          shape=value.shape,
          dtype=value.dtype,
          data=value.data,
          layout=value.layout)
    return nndct_tensor


class NodeConvertor(object):
  r"""convert raw_node to nndct_node"""

  def __call__(self, parser, raw_node, node_scope=''):
    nndct_node = Node(
        name=get_full_name(node_scope, raw_node.name),
        dtype=convert_dtype(raw_node.dtype),
        idx=raw_node.idx)

    nndct_node.raw_kind = raw_node.kind
    nndct_node.schema = raw_node.schema
    nndct_node.is_custom_extension = raw_node.is_custom_pyop
    nndct_node.caller = raw_node.pyobj
    
    blob_tensor_convertor = TensorConvertor()
    for op in raw_node.outputs:
      nndct_tensor = blob_tensor_convertor(node_scope, op)
      nndct_tensor.node = nndct_node
      nndct_node.out_tensors.append(nndct_tensor)
      parser.visited_blob_tensors[op.name] = nndct_tensor

    for ip in raw_node.flatten_inputs:
      if ip.name in parser.visited_blob_tensors:
        nndct_node.in_tensors.append(parser.visited_blob_tensors[ip.name])
      elif ip.name in parser.visited_param_tensors:
        parser.node_params[nndct_node].append(
            parser.visited_param_tensors[ip.name])
        nndct_node.in_tensors.append(parser.visited_param_tensors[ip.name])

    if not raw_node.inputs:
      parser.node_input_args[nndct_node].append(nndct_node.out_tensors[0])
    else:
      parser.node_input_args[nndct_node].extend(
          [parser.get_nndct_value(i) for i in raw_node.inputs])

    return nndct_node


class OpCreator(object):

  op_convert_map = {
    "__getitem__": "index",
    "TupleUnpack": "tuple_unpack",
    "ListUnpack": "tuple_unpack",
    "__derive_index": "derive_loop_index",
    "_cast_Float": "cast_float",
    "_cast_Int": "cast_int",
    "add_": "add"
  }
  def __init__(self, device_type):
    self._device_type = device_type

  def __call__(self, parser, nndct_node):
    self.cur_node = nndct_node
    op_type = self.op_convert_map.get(nndct_node.raw_kind, nndct_node.raw_kind)
    if hasattr(self, op_type):
      op = getattr(self, op_type)(*parser.node_input_args[nndct_node])
    elif nndct_node.is_custom_extension:
      op = self.custom_op(nndct_node, *parser.node_input_args[nndct_node])
    else:
      op = self.default(nndct_node, *parser.node_input_args[nndct_node])

    del self.cur_node
    del nndct_node.schema
    return op
  

  # torch function should always add 'input' config
  # nn.function, nn.Module and torch.Tensor can ignore 'input'

  def Param(self, *args):
    op = TorchUnaryOp(NNDCT_OP.INPUT, "input", force_to_primitive=True)
    input_name = f"args[{args[0].node.name.split('_')[-1]}]"
    op.set_config("input", input_name)
    return op

  def _convolution(self, input, weight, bias, stride, padding, dilation,
                   transposed, output_padding, groups, benchmark, deterministic,
                   cudnn_enabled, *args):
    # weight_size = weight.type().sizes()
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
          op = TorchConv1d("DepthwithConv1D is unsupported")
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

  def batch_norm(self, input, weight, bias, running_mean, running_var, training,
                 momentum, eps, cudnn_enabled):
    op = TorchBatchNorm(input.ndim)
    op.set_param(op.ParamName.GAMMA, weight)
    op.set_param(op.ParamName.BETA, bias)
    op.set_param(op.ParamName.MOVING_MEAN, running_mean)
    op.set_param(op.ParamName.MOVING_VAR, running_var)
    weight_size = weight.shape
    if training:
      op.set_config('is_training', True)
    op.set_config('num_features', weight_size[0])
    op.set_config('eps', eps)
    op.set_config('momentum', momentum)
    op.set_attr(op.AttrName.SCALE, True)
    op.set_attr(op.AttrName.CENTER, True)

    dims2axis_map = {
        2: 1,  # NC -> 1
        3: 1,  # NCL -> 1
        4: 3,  # NHWC -> 3
        5: 1  # NCDHW -> 1
    }
    op.set_attr(op.AttrName.AXIS, dims2axis_map[len(input.shape)])
    return op

  @staticmethod
  def _max_pool1d(op, input, kernel_size, stride, padding, dilation, ceil_mode):
    op.set_config('kernel_size', list(kernel_size))
    if not stride:
      op.set_config('stride', list(kernel_size))
    else:
      op.set_config('stride', list(stride))

    if ceil_mode:
      op.set_config('ceil_mode', True)
    else:
      op.set_config('ceil_mode', False)

    op.set_config('padding', list(padding))
    op.set_config('dilation', list(dilation))

    return op

  def max_pool1d(self, *args):
    op = TorchMaxPool1d()
    return self._max_pool1d(op, *args)

  def max_pool1d_with_indices(self, *args):
    op = TorchMaxPool1d()
    op.set_config("return_indices", True)
    return self._max_pool1d(op, *args)

  @staticmethod
  def _max_pool2d(op, input, kernel_size, stride, padding, dilation, ceil_mode):
    op.set_config('kernel_size', list(kernel_size))
    if not stride:
      op.set_config('stride', list(kernel_size))
    else:
      op.set_config('stride', list(stride))

    if ceil_mode:
      op.set_config('ceil_mode', True)
    else:
      op.set_config('ceil_mode', False)

    op.set_config('padding', list(padding))
    op.set_config('dilation', list(dilation))

    return op

  def max_pool2d(self, *args):
    op = TorchMaxPool()
    return self._max_pool2d(op, *args)

  def max_pool2d_with_indices(self, *args):
    op = TorchMaxPool()
    op.set_config("return_indices", True)
    return self._max_pool2d(op, *args)

  def avg_pool2d(self,
                 input,
                 kernel_size,
                 stride,
                 padding,
                 ceil_mode,
                 count_include_pad,
                 divisor_override=None):

    op = TorchAvgPool()

    op.set_config('kernel_size', list(kernel_size))
    if not stride:
      op.set_config('stride', list(kernel_size))
    else:
      op.set_config('stride', list(stride))

    if ceil_mode:
      op.set_config('ceil_mode', True)
    else:
      op.set_config('ceil_mode', False)

    if count_include_pad:
      op.set_config('count_include_pad', True)
    else:
      op.set_config('count_include_pad', False)

    op.set_config('padding', list(padding))

    return op

  def adaptive_avg_pool2d(self, input, output_size):
    if output_size == 1 or (isinstance(output_size, (tuple, list)) and
                            tuple(output_size) == (1, 1)):
      op = TorchAdaptiveAvgPool()
      op.set_attr(op.AttrName.GLOBAL, True)
      op.set_attr(op.AttrName.PAD_MODE, 0)
      op.set_attr(op.AttrName.COUNT_INCLUDE_PAD, True)
      op.set_attr(op.AttrName.PAD, [0, 0, 0, 0])
      op.set_config('output_size', 1)
    else:
      op = TorchAvgPool()
      op.set_attr(op.AttrName.GLOBAL, False)
      input_size = input.shape[2:4]
      stride_h = math.floor(input_size[0] / output_size[0])
      stride_w = math.floor(input_size[1] / output_size[1])
      kernel_h = input_size[0] - (output_size[0] - 1) * stride_h
      kernel_w = input_size[1] - (output_size[1] - 1) * stride_w
      op.set_config('kernel_size', [kernel_h, kernel_w])
      op.set_config('stride', [stride_h, stride_w])
      op.set_config('ceil_mode', False)
      op.set_config('count_include_pad', True)
      op.set_config('padding', [0, 0])

    return op

  def addmm(self, bias, input, weight, beta=None, alpha=None):
    if (bias.node is not None) or (weight.node is not None):
      op = TorchBaseOperation(NNDCT_OP.ADDMM, "addmm")
      op.set_config("input", bias)
      op.set_config("mat1", input)
      op.set_config("mat2", weight)
    else:
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
  
  def linear(self, input, weight, bias):
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

  def flatten(self, input, start_dim=0, end_dim=-1):
    """TODO(wluo@xilinx.com): need to get the input shape, then
    we can get the correct end_dim 
    """
    op = TorchFlatten()
    op.set_config('input', input)

    if end_dim == -1:
      end_dim = 3
    op.set_config("start_dim", start_dim)
    op.set_config("end_dim", end_dim)
    return op

  def relu_(self, input):
    op = TorchReLU()
    op.set_config('inplace', True)
    return op

  def relu(self, input):
    op = TorchReLU()
    op.set_config('inplace', False)
    return op

  def leaky_relu_(self, input, negative_slope=0.01):
    op = TorchLeakyReLU()
    op.set_config("negative_slope", negative_slope)
    op.set_config("inplace", True)
    return op

  def leaky_relu(self, input, negative_slope=0.01):
    op = TorchLeakyReLU()
    op.set_config("negative_slope", negative_slope)
    op.set_config("inplace", False)
    return op

  def add(self, input, other, alpha=None):
    if (isinstance(input, Tensor) and input.is_complete_tensor()) \
     or (isinstance(other, Tensor) and other.is_complete_tensor()):
      op = TorchAdd()
    elif isinstance(input, list) or isinstance(other, list):
      op = TorchBaseOperation(
          NNDCT_OP.LIST_ADD, NNDCT_OP.LIST_ADD, force_to_primitive=False)
    else:
      op = TorchBinaryOp(NNDCT_OP.SCALAR_ADD, "add")

    op.set_config('input', input)
    op.set_config('other', other)
    if alpha is not None:
      op.set_config('alpha', alpha)
    return op

  def size(self, input, dim=None):
    op = TorchSize(input.ndim)
    if dim is not None:
      op.set_config("dim", dim)
    return op

  def view(self, input, shape):
    cur_node = self.cur_node
    op = TorchView()
    op.set_config("input", input)
    op.set_config("shape", shape)
    return op

  def reshape(self, input, shape):
    return self.view(input, shape)

  def dropout(self, input, p, train):
    op = TorchDropout()
    if train:
      op.set_config("p", p)
    op.set_config("inplace", True)
    return op

  def dropout_(self, input, p, train):
    return self.dropout(input, p, train)

  def cat(self, tensors, dim):
    in_ndim = tensors[0].ndim if isinstance(tensors, (tuple, list)) else tensors.ndim
    op = TorchCat(in_ndim)
    op.set_config("dim", dim)
    op.set_config("tensors", tensors)
    return op

  def mean(self, input, dim=None, keepdim=False, dtype=None):
    op = TorchPermuteInvarOp(len(input.shape), NNDCT_OP.MEAN, "mean")
    op.set_config("input", input)
    op.set_config("dim", dim)
    op.set_config("keepdim", keepdim)
    return op

  def relu6(self, input, inplace):
    op = TorchUnaryOp(NNDCT_OP.RELU6, "ReLU6")
    op.set_config('input', input)
    op.set_config('inplace', inplace)
    return op

  def hardtanh(self, input, min_val, max_val):
    if min_val == 0.0 and max_val == 6.0:
      return self.relu6(input, False)
    else:
      op = TorchHardTanh()
      op.set_config("min_val", min_val)
      op.set_config("max_val", max_val)
      op.set_config("inplace", False)
      return op

  def hardtanh_(self, input, min_val, max_val):
    if min_val == 0.0 and max_val == 6.0:
      return self.relu6(input, True)
    else:
      op = TorchHardTanh()
      op.set_config("min_val", min_val)
      op.set_config("max_val", max_val)
      op.set_config("inplace", True)
      return op

  def transpose(self, input, dim0, dim1):
    op = TorchTranspose(input.ndim)
    op.set_config("input", input)
    op.set_config("dim0", dim0)
    op.set_config("dim1", dim1)
   

    return op

  def contiguous(self, *args):
    op = TorchContiguous()
    return op

  def ConstantChunk(self, input, chunks, dim):
    return self.chunk(input, chunks, dim)

  def chunk(self, input, chunks, dim):
    op = TorchChunk()
    op.set_config("input", input)
    op.set_config("chunks", chunks)
    op.set_config("dim", dim)
    return op

  def _interpolate(self,
                   input,
                   size=None,
                   scale_factor=None,
                   mode="'nearest'",
                   align_corners=None,
                   recompute_scale_factor=None):
    
    if input.ndim == 4:
      if mode == "'nearest'":
        op = TorchInterpolate(len(input.shape))
      elif mode == "'bilinear'":
        op = TorchResizeLinear(len(input.shape))
    elif input.ndim == 5:
      if mode == "'trilinear'":
        op = TorchResizeTrilinear(input.ndim)
      elif mode == "'nearest'":
        op = TorchBaseOperation(NNDCT_OP.RESIZE_NEAREST_3D, "interpolate")
        
  
    op.set_config("input", input)
    op.set_config("mode", mode)
    if size is not None:
      op.set_config("size", size)
    if align_corners is not None:
      op.set_config("align_corners", bool(align_corners))
    if size is None and scale_factor is not None:
      op.set_config("scale_factor", scale_factor)

    if recompute_scale_factor is not None:
      op.set_config("recompute_scale_factor", bool(recompute_scale_factor))
    return op

  def upsample_bilinear2d(self,
                          input,
                          tensor_list,
                          align_corners,
                          scale_h=None,
                          scale_w=None):
    if scale_w is not None and scale_h != scale_w:
      scale = [scale_h, scale_w]
    else:
      scale = scale_h
    return self._interpolate(
        input,
        size=tensor_list,
        scale_factor=scale,
        mode="'bilinear'",
        align_corners=align_corners)

  def upsample_nearest2d(self, input, tensor_list, scale_h=None, scale_w=None):
    if scale_w is not None and scale_h != scale_w:
      scale = [scale_h, scale_w]
    else:
      scale = scale_h
    return self._interpolate(input, size=tensor_list, scale_factor=scale)

  def upsample_trilinear3d(self, input, tensor_list, align_corners, scale_factor_d=None, scale_factor_h=None, scale_factor_w=None):
    if scale_factor_h is not None and scale_factor_w is not None:
      scale = [scale_factor_d, scale_factor_h, scale_factor_w]
    else:
      scale = scale_factor_d
    return self._interpolate(
        input,
        size=tensor_list,
        scale_factor=scale,
        mode="'trilinear'",
        align_corners=align_corners)
  
  def upsample_nearest3d(self, input, output_size, scale_factor=None):
    return self._interpolate(
        input,
        size=output_size,
        scale_factor=scale_factor)
    
  def NumToTensor(self, input, *args):
    op = TorchTensor()
    op.set_config("data", input)
    op.set_config("dtype", input.dtype)
    op.set_config("device", f"'{self._device_type}'")
    return op

  def Constant(self, tensor, *args):
    op = TorchConst()
    if not tensor.is_complete_tensor():
      if isinstance(tensor.data, np.ndarray):
          tensor.from_ndarray(tensor.data)
      else:
        np_data = np.array(tensor.data, dtype=np.float32)
        tensor.from_ndarray(np_data)
      
    op.set_config('data', tensor.data.tolist())
    op.set_config('dtype', convert_np_type_to_pytorch_type(tensor.dtype))
    op.set_config('device', f"'{self._device_type}'")
    return op

  def mul(self, input, other):
    if (isinstance(input, Tensor) and input.is_complete_tensor()) \
     or (isinstance(other, Tensor) and other.is_complete_tensor()):
      op = TorchMul()
    else:
      op = TorchBinaryOp(NNDCT_OP.SCALAR_MUL, "mul")

    op.set_config("input", input)
    op.set_config("other", other)
    return op

  def to(self, input, *args):
    # op = TorchCast()
    op = TorchUnaryOp(NNDCT_OP.CAST, "to")
    op.set_config("input", input)
    if isinstance(args[0], str):
      op.set_config('dtype', scalar_type_to_pytorch_type[args[1]])
    else:
      op.set_config('dtype', scalar_type_to_pytorch_type[args[0]])

    return op

  def floor(self, input):
    # op = TorchFloor()
    op = TorchUnaryOp(NNDCT_OP.FLOOR, "floor")
    op.set_config("input", input)
    return op

  def Int(self, *args):
    op = TorchUnaryOp(NNDCT_OP.INT, "int", force_to_primitive=True)
    op.set_config("input", args[0])
    return op

  def permute(self, input, dims):
    op = TorchPermute(len(input.shape))
    op.set_config("dims", dims)
    op.set_config("input", input)
    # op.get_attr(op.AttrName.ORDER)
    return op

  def floor_divide(self, input, other):
    op = TorchFloorDiv()
    op.set_config("input", input)
    op.set_config("other", other)
    return op

  def div(self, input, other):
    if python_dtype(input) == "int" and python_dtype(other) == "int":
      op = TorchBinaryOp(NNDCT_OP.FLOOR_DIV, "//", force_to_primitive=False)
    else:
      op = TorchBinaryOp(NNDCT_OP.DIV, "div")
      
    # if python_dtype(input) == "float" or python_dtype(other) == "float":
    #   op = TorchBinaryOp(NNDCT_OP.DIV, "div")
    # else:
    #   op = TorchBinaryOp(NNDCT_OP.FLOOR_DIV, "//", force_to_primitive=False)

    op.set_config("input", input)
    op.set_config("other", other)
    return op

  def softmax(self, input, dim, dtype=None):
    op = TorchSoftmax(len(input.shape))
    op.set_config("dim", dim)
    return op

  def sigmoid(self, input):
    op = TorchSigmoid()
    return op

  def hardswish(self, input):
    op = TorchUnaryOp(NNDCT_OP.HSWISH, "Hardswish")
    op.set_config('inplace', False)
    op.set_config('input', input)
    return op

  def hardswish_(self, input):
    op = TorchUnaryOp(NNDCT_OP.HSWISH, "Hardswish")
    op.set_config('inplace', True)
    op.set_config('input', input)
    return op

  def hardsigmoid(self, input):
    op = TorchUnaryOp(NNDCT_OP.HSIGMOID, "Hardsigmoid")
    op.set_config('inplace', False)
    op.set_config('input', input)
    return op

  def hardsigmoid_(self, input):
    op = TorchUnaryOp(NNDCT_OP.HSIGMOID, "Hardsigmoid")
    op.set_config('inplace', True)
    op.set_config('input', input)
    return op

  def strided_slice(self, input, dim, start, end, step):
    assert input.ndim is not None
    op = TorchSlice(input.ndim)
    begin = [0] * input.ndim
    last = [NNDCT_CONSTANT.INT_MAX] * input.ndim
    stride = [1] * input.ndim
    #op = TorchSlice(len(input.shape))
    #begin = [0] * len(input.shape)
    #last = [NNDCT_CONSTANT.INT_MAX] * len(input.shape)
    #stride = [1] * len(input.shape)
    for i, pos in enumerate(dim):
      begin[pos] = start[i]
      if isinstance(end[i], Tensor) or (isinstance(end[i], int) and
                                        end[i] < last[pos]):
        last[pos] = end[i]

      stride[pos] = step[i]
    op.set_config("input", input)
    op.set_config("start", begin)
    op.set_config("end", last)
    op.set_config("step", stride)
    return op

  def sub(self, input, other, alpha):
    if (isinstance(input, Tensor) and input.is_complete_tensor()) \
     or (isinstance(other, Tensor) and other.is_complete_tensor()):
      op = TorchSub()
    else:
      op = TorchBinaryOp(NNDCT_OP.SCALAR_SUB, "sub")
    # op = TorchSub()
    op.set_config("input", input)
    op.set_config("other", other)
    op.set_config("alpha", alpha)
    return op

  def rsub(self, input, other, alpha):
    op = TorchRsub()
    op.set_config("input", other)
    op.set_config("other", input)
    op.set_config("alpha", alpha)
    return op

  def exp(self, input, *args):
    op = TorchExp()
    op.set_config("input", input)
    return op

  def detach(self, input, *args):
    op = TorchDetach()
    op.set_config("input", input)
    return op

  def select(self, input, dim, index):
    op = TorchSelect()
    op.set_config("input", input)
    op.set_config("dim", dim)
    op.set_config("index", index)
    op.set_attr_by_name("dim", dim)
    op.set_attr_by_name("index", index)
    return op

  def repeat(self, input, sizes):
    op = TorchRepeat()
    op.set_config('repeats', sizes)
    return op

  def copy_(self, input, src, non_blocking):
    op = TorchInplaceCopy()
    op.set_config('other', src)
    op.set_config('non_blocking', bool(non_blocking))
    return op

  def expand(self, input, size, *args):
    op =  TorchBaseOperation(NNDCT_OP.EXPAND, "expand", force_to_primitive=True)
    op.set_config('input', input)
    op.set_config('size', size)
    return op

  def t(self, input, *args):
    return self.transpose(input, 0, 1)

  def empty(self, *args):
    op = TorchEmpty()
    op.set_config('size', args[0])
    op.set_config('dtype', scalar_type_to_pytorch_type[args[1]])
    op.set_config('device', f"'{self._device_type}'")
    return op

  
  def unsqueeze(self, input, dim):
    op = TorchUnsqueeze()
    op.set_config("input", input)
    op.set_config("dim", dim)
    return op

  def lstm(self, *args):
    if isinstance(args[3], list):
      raise NotImplementedError('Unimplement packed lstm')
    else:
      return self._lstm_full(*args)

  def _lstm_full(self, input, hidden, weight_v, has_biases, num_layers, dropout,
                 train, bidirectional, batch_first):
    op = TorchLstm()
    op.set_config('bias', bool(has_biases))
    op.set_config('bidirectional', bool(bidirectional))
    op.set_config('input_size', input.shape[-1])
    op.set_config('hidden_size', hidden[0].shape[-1])
    op.set_config('batch_first', bool(batch_first))
    op.set_config('num_layers', num_layers)
    op.set_config('dropout', dropout)
    weight_ih = []
    weight_hh = []
    op.set_param(op.ParamName.WEIGHT_IH, weight_ih)
    op.set_param(op.ParamName.WEIGHT_HH, weight_hh)
    layer_offset = 2

    if bool(has_biases):
      bias = []
      op.set_param(op.ParamName.BIAS, bias)
      layer_offset += 2

    if bool(bidirectional):
      weight_ih_reverse = []
      weight_hh_reverse = []
      op.set_param(op.ParamName.WEIGHT_IH_REVERSE, weight_ih_reverse)
      op.set_param(op.ParamName.WEIGHT_HH_REVERSE, weight_hh_reverse)
      layer_offset += 2
      if bool(has_biases):
        bias_reverse = []
        op.set_param(op.ParamName.BIAS_REVERSE, bias_reverse)
        layer_offset += 2

    for layer_no in range(num_layers):
      i = 0
      for param_name, param in op.params.items():
        step = layer_no * layer_offset
        if param_name in [op.ParamName.BIAS, op.ParamName.BIAS_REVERSE]:
          param.extend([weight_v[i + step], weight_v[i + 1 + step]])
          i += 2
        else:
          param.append(weight_v[i + step])
          i += 1

    return op

  def gru(self, *args):
    if isinstance(args[3], list):
      raise NotImplementedError('Unimplement packed gri')
    else:
      return self._gru_full(*args)

  def _gru_full(self, input, hidden, weight_v, has_biases, num_layers, dropout,
                train, bidirectional, batch_first):
    op = TorchGru()
    op.set_config('bias', bool(has_biases))
    op.set_config('bidirectional', bool(bidirectional))
    op.set_config('input_size', input.shape[-1])
    op.set_config('hidden_size', hidden.shape[-1])
    op.set_config('batch_first', bool(batch_first))
    op.set_config('num_layers', num_layers)
    op.set_config('dropout', dropout)

    weight_ih = []
    weight_hh = []
    op.set_param(op.ParamName.WEIGHT_IH, weight_ih)
    op.set_param(op.ParamName.WEIGHT_HH, weight_hh)
    layer_offset = 2

    if bool(has_biases):
      bias_ih = []
      bias_hh = []
      op.set_param(op.ParamName.BIAS_IH, bias_ih)
      op.set_param(op.ParamName.BIAS_HH, bias_hh)
      layer_offset += 2

    if bool(bidirectional):
      weight_ih_reverse = []
      weight_hh_reverse = []
      op.set_param(op.ParamName.WEIGHT_IH_REVERSE, weight_ih_reverse)
      op.set_param(op.ParamName.WEIGHT_HH_REVERSE, weight_hh_reverse)
      layer_offset += 2
      if bool(has_biases):
        bias_ih_reverse = []
        bias_hh_reverse = []
        op.set_param(op.ParamName.BIAS_IH_REVERSE, bias_ih_reverse)
        op.set_param(op.ParamName.BIAS_HH_REVERSE, bias_hh_reverse)
        layer_offset += 2

    for layer_no in range(num_layers):
      i = 0
      for param_name, param in op.params.items():
        step = layer_no * layer_offset
        param.append(weight_v[i + step])
        i += 1

    return op

  def zeros(self, sizes, dtype, layout, device, pin_memory=False):
    op = TorchZeros()
    op.set_config('size', sizes)
    dtype = 6 if dtype is None else dtype
    op.set_config('dtype', scalar_type_to_pytorch_type[dtype])
    op.set_config('device', f"'{self._device_type}'")
    return op

  def constant_pad_nd(self, input, pad, value):
    op = TorchPad()
    op.set_config('input', input)
    op.set_config('pad', pad)
    op.set_config('value', value)
    op.set_config('mode', "'constant'")
    return op

  def replication_pad2d(self, input, pad):
    for pad_len in pad:
      if not isinstance(pad_len, Tensor):
        if pad_len > 1:
          raise ValueError(
              "Only support pad 1 row or 1 col in replication mode")

    op = TorchPad()
    op.set_config('input', input)
    op.set_config('pad', pad)
    op.set_config('value', 0.0)
    op.set_config('mode', "'replicate'")
    return op

  def matmul(self, input, other, *args):
    if "weight" in other.name:
      op = TorchLinear()
      op.set_param(op.ParamName.WEIGHTS, other)
      op.set_config("bias", False)
      op.set_config("out_features", other.shape[0])
      op.set_config("in_features", other.shape[1])
      return op
    else:
      op = TorchMatmul()
      op.set_config("input", input)
      op.set_config("other", other)
      return op

  def clamp(self, input, min, max):
    op = TorchClamp()
    op.set_config("input", input)
    op.set_config("min", min)
    op.set_config("max", max)
    op.set_attr_by_name("min", min)
    op.set_attr_by_name("max", max)
    return op

  def clamp_min(self, input, min):
    op = TorchClamp()
    op.set_config("input", input)
    op.set_config("min", min)
    op.set_attr_by_name("min", min)
    return op

  def tanh(self, input, *args):
    op = TorchTanh()
    return op

  def arange(self, *args):
    op = TorchArange()

    if len(args) == 5:
      # aten::arange(Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
      op.set_config("end", args[0])
      #TODO: should inter dtype frome end/start/step type
      op.set_config("dtype", scalar_type_to_pytorch_type[4])
      op.set_config("device", f"'{self._device_type}'")
    elif len(args) == 6:
      # aten::arange(Scalar start, Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
      op.set_config("start", args[0])
      op.set_config("end", args[2])
      op.set_config("dtype", scalar_type_to_pytorch_type[4])
      op.set_config("device", f"'{self._device_type}'")
    elif len(args) == 7:
      # aten::arange(Scalar start, Scalar end, Scalar step, ScalarType dtype, Layout, Device, bool pin_memory)
      op.set_config("start", args[0])
      op.set_config("end", args[1])
      op.set_config("step", args[2])
      dtype = scalar_type_to_pytorch_type[
          args[3]] if args[3] is not None else scalar_type_to_pytorch_type[4]
      op.set_config("dtype", dtype)
      op.set_config("device", f"'{self._device_type}'")
    else:
      raise NotImplementedError("Unknown aten::arange signature taking " +
                                str(len(args)) + " arguments.")
    return op

  def slice_tensor_inplace_copy(self, input, src, non_blocking, dim, index):
    op = TorchBaseOperation(
        NNDCT_OP.SLICE_TENSOR_INPLACE_COPY,
        NNDCT_OP.SLICE_TENSOR_INPLACE_COPY,
        force_to_primitive=True)
    op.set_config("input", input)
    op.set_config("source", src)
    op.set_config("dim", dim)
    op.set_config("index", index)
    return op

  def norm(self, input, p, dim, keepdim):
    assert p == 2, "Only Support L2 norm"
    op = TorchBaseOperation(NNDCT_OP.NORM, "norm")
    op.set_config("input", input)
    op.set_config("dim", dim)
    op.set_config("keepdim", bool(keepdim))
    op.set_config("p", p)
    op.set_attr_by_name("dim", dim)
    op.set_attr_by_name("keepdim", bool(keepdim))
    op.set_attr_by_name("p", p)
    return op

  def expand_as(self, input, other):
    op = TorchBaseOperation(NNDCT_OP.EXPAND_AS, "expand_as", force_to_primitive=True)
    op.set_config("input", input)
    op.set_config("other", other)
    return op

  def max(self, input, dim, keepdim):
    op = TorchPermuteInvarOp(input.ndim, NNDCT_OP.MAX, "max")
    # op = TorchBaseOperation(NNDCT_OP.MAX, "max")
    op.set_config("input", input)
    op.set_config("dim", dim)
    op.set_config("keepdim", bool(keepdim))
    return op

  def min(self, input, dim, keepdim):
    op = TorchPermuteInvarOp(len(input.shape), NNDCT_OP.MIN, "min")
    op.set_config("input", input)
    op.set_config("dim", dim)
    op.set_config("keepdim", bool(keepdim))
    return op

  def squeeze(self, input, dim=None):
    # op = TorchBaseOperation(NNDCT_OP.SQUEEZE, "squeeze")
    op = TorchSqueeze(len(input.shape))
    op.set_config("input", input)
    if dim is not None:
      if dim == -1:
        dim = len(input.shape) - 1
      op.set_config("dim", dim)
    return op

  def eq(self, input, other):
    #op = TorchBaseOperation(NNDCT_OP.EQUAL, "eq")
    if (isinstance(input, Tensor) and input.is_complete_tensor()) \
    or (isinstance(other, Tensor) and other.is_complete_tensor()):
      op = TorchBaseOperation(NNDCT_OP.EQUAL, "eq")
    else:
      op = TorchBaseOperation(NNDCT_OP.SCALAR_EQUAL, NNDCT_OP.SCALAR_EQUAL, force_to_primitive=False)
    op.set_config("input", input)
    op.set_config("other", other)
    return op

  def index(self, input, index):
    op = TorchBaseOperation(
        NNDCT_OP.INDEX, NNDCT_OP.INDEX, force_to_primitive=True)
    op.set_config("input", input)
    op.set_config("index", index)
    return op

  def index_put_(self, input, indices, value, accumulate, *args):
    op = TorchBaseOperation(
        NNDCT_OP.INDEX_INPUT_INPLACE,
        NNDCT_OP.INDEX_INPUT_INPLACE,
        force_to_primitive=False)
    op.set_config("input", input)
    op.set_config("indices", indices)
    # op.set_config("indices", [index if index is not None else ":" for index in indices])
    op.set_config("values", value)
    op.set_config("accumulate", bool(accumulate))
    return op

  def stack(self, input, dim):
    op = TorchBaseOperation(NNDCT_OP.STACK, "stack")
    op.set_config("tensors", input)
    op.set_config("dim", dim)
    return op

  def bmm(self, input, mat2):
    op = TorchBaseOperation(NNDCT_OP.BMM, "bmm")
    op.set_config("input", input)
    op.set_config("mat2", mat2)
    return op

  def embedding_bag(self,
                    weight,
                    indices,
                    offsets,
                    scale_grad_by_freq,
                    mode,
                    sparse,
                    per_sample_weights=None):
    mode_map = {0: "sum", 1: "mean", 2: "max"}

    op = TorchEmbeddingBag()
    op.set_param(op.ParamName.WEIGHT, weight)

    op.set_config("num_embeddings", weight.shape[0])
    op.set_config("embedding_dim", weight.shape[1])
    op.set_config("scale_grad_by_freq", bool(scale_grad_by_freq))
    op.set_config("mode", f"'{mode_map[mode]}'")
    op.set_config("sparse", bool(sparse))
    return op

  def feature_dropout(self, *args):
    return self.dropout(*args)

  def QuantStubF(self, input, *args):
    op = TorchUnaryOp(
        NNDCT_OP.QUANT_STUB, "quant_input", force_to_primitive=True)
    op.set_config("input", input)
    return op

  def DeQuantStubF(self, input, *args):
    op = TorchUnaryOp(NNDCT_OP.DEQUANT_STUB, "dequant_output", force_to_primitive=True)
    op.set_config("input", input)
    return op

  def RelukF(self, input, inplace, channel_max=6.0):
    op = TorchBaseOperation(NNDCT_OP.RELUK, "Reluk", force_to_primitive=True)
    op.set_config("input", input)
    op.set_config("channel_max", channel_max)
    return op
  
  def ChannelScaleF(self, input, inplace, channel_scale=1.0):
    op = TorchBaseOperation(NNDCT_OP.CHANNEL_SCALE, "Channel_Scale", force_to_primitive=True)
    op.set_config("input", input)
    op.set_config("channel_scale", channel_scale)
    return op

  def strided_slice_inplace_copy(self, des, dim, start, end, step, src):
    op = TorchBaseOperation(
        NNDCT_OP.STRIDED_SLICE_INPLACE_COPY,
        NNDCT_OP.STRIDED_SLICE_INPLACE_COPY,
        force_to_primitive=False)
    
    begin = [0] * des.ndim
    last = [NNDCT_CONSTANT.INT_MAX] * des.ndim
    stride = [1] * des.ndim
    for i, pos in enumerate(dim):
      begin[pos] = start[i]
      stride[pos] = step[i]
      last[pos] = end[i] 
      
    op.set_config("destination", des)
    op.set_config("source", src)
    op.set_config("start", begin)
    op.set_config("end", last)
    op.set_config("step", stride)
    return op

  def pixel_shuffle(self, input, upscale_factor):
    op = TorchPixelShuffle()
    op.set_config("upscale_factor", upscale_factor)
    op.set_attr(op.AttrName.UPSCALE, True)
    return op
  
  def pixel_unshuffle(self, input, downscale_factor):
    op = TorchPixelUnshuffle()
    op.set_config("downscale_factor", downscale_factor)
    op.set_attr(op.AttrName.UPSCALE, False)
    return op

  def Loop(self, max_trip_count, initial_condition, *args):
    op = TorchBaseOperation(
        NNDCT_OP.LOOP, NNDCT_OP.LOOP, force_to_primitive=False)
    if isinstance(max_trip_count, int) and max_trip_count == sys.maxsize:
      op.set_config("is_while_loop", True)
    else:
      op.set_config("is_while_loop", False)

      
    op.set_config("max_trip_count", max_trip_count)
    if isinstance(initial_condition, int):
      op.set_config("initial_condition", bool(initial_condition))
    else:
      op.set_config("initial_condition", initial_condition)

    initial_vars = [arg for arg in args]
    op.set_config("initial_loop_vars", initial_vars)

    return op

  def mm(self, input, other):
    op = TorchMatmul()
    op.set_config("input", input)
    op.set_config("other", other)
    return op

  def layer_norm(self, input, normalized_shape, weight, bias, eps,
                 cudnn_enabled):
    op = TorchLayerNorm()
    op.set_param(op.ParamName.GAMMA, weight)
    op.set_param(op.ParamName.BETA, bias)
    op.set_config("normalized_shape", normalized_shape)
    op.set_config("eps", eps)
    return op

  def new_zeros(self, input, *args):
    return self.zeros(*args)

  def neg(self, input):
    op = TorchUnaryOp(NNDCT_OP.NEG, "neg")
    op.set_config("input", input)
    return op

  def grid_sampler(self, input, grid, mode, padding_mode, align_corners):
    mode_map = {
        0: "bilinear",
        1: "nearest",
    }
    padding_mod_map = {0: "zeros", 1: "border", 2: "reflection"}

    op = TorchBaseOperation(NNDCT_OP.GRID_SAMPLE, "grid_sample")
    op.set_config("input", input)
    op.set_config("grid", grid)
    op.set_config("mode", f"'{mode_map[mode]}'")
    op.set_config("padding_mode", f"'{padding_mod_map[padding_mode]}'")
    op.set_config("align_corners", bool(align_corners))
    return op


  def sum(self, input, dim, keepdim, dtype=None):
    op = TorchPermuteInvarOp(input.ndim, NNDCT_OP.SUM, "sum")
    op.set_config("input", input)
    op.set_config("dim", dim)
    op.set_config("keepdim", bool(keepdim))
    return op
  
  
  def tuple_unpack(self, input):
    op = TorchBaseOperation(NNDCT_OP.TUPLE_UNPACK, NNDCT_OP.TUPLE_UNPACK, force_to_primitive=False)
    op.set_config("input", input)
    return op

   
  def derive_loop_index(self, index, start, step):
    op = TorchBaseOperation(NNDCT_OP.DERIVE_LOOP_INDEX, NNDCT_OP.DERIVE_LOOP_INDEX, force_to_primitive=False)
    op.set_config("input", index)
    op.set_config("start", start)
    op.set_config("step", step)
    return op
    
  
  def cast_float(self, input, non_blocking):
    return self.to(input, 6)

  def cast_int(self, input, non_blocking):
    return self.to(input, 3)

  def Bool(self, input):
    return self.to(input, 11)

  def ceil(self, input):
    op = TorchBaseOperation(NNDCT_OP.CEIL, 'ceil')
    op.set_config("input", input)
    return op
 
  def slice(self, input, dim, start, end, step):
    op = TorchBaseOperation(NNDCT_OP.SLICE, NNDCT_OP.SLICE, force_to_primitive=False)
    op.set_config("input", input)
    op.set_config("dim", dim)
    op.set_config("start", start)
    op.set_config("end", end)
    op.set_config("step", step)
    return op

  def len(self, input):
    op = TorchBaseOperation(NNDCT_OP.LENGTH, NNDCT_OP.LENGTH, force_to_primitive=False)
    op.set_config("input", input)
    return op

  def lt(self, input, other):
    op = TorchBaseOperation(NNDCT_OP.SCALAR_LESS_THAN, NNDCT_OP.SCALAR_LESS_THAN, force_to_primitive=False)
    op.set_config("input", input)
    op.set_config("other", other)
    return op
    
  def If(self, condition):
    op = TorchBaseOperation(NNDCT_OP.IF, NNDCT_OP.IF, force_to_primitive=False)
    op.set_config("condition", condition)
    return op

  def tensor(self, data, dtype, device, required_grad=False):
    op = TorchTensor()
    op.set_config("data", data)
    if dtype is not None:
      op.set_config("dtype", scalar_type_to_pytorch_type(dtype))
    op.set_config("device", f"'{self._device_type}'")
    return op
    
  def embedding(self, weight, indices, padding_idx, scale_grad_freq, sparse):
    op = TorchEmbedding()
    op.set_config("num_embeddings", weight.shape[0])
    op.set_config("embedding_dim", weight.shape[1])
    op.set_config("padding_idx", padding_idx)
    op.set_param(op.ParamName.WEIGHT, weight)
    return op

  def item(self, input):
    op = TorchBaseOperation(NNDCT_OP.TENSOR_TO_SCALAR, 'item')
    return op

    
  def default(self, node, *args):
    schema2torchop = GLOBAL_MAP.get_ele(NNDCT_KEYS.TORCH_SCHEMA_OP_TABLE)
    schema_handler = SchemaHelper(node.schema)
    torchop = schema2torchop.get(schema_handler.toString(), None)
    if torchop is None:
      op = TorchUnknownOperation(node.raw_kind)
      return op
    node2caller = GLOBAL_MAP.get_ele(NNDCT_KEYS.NODE_CALLER_MAP)
    if node2caller is None:
      node2caller: Dict[str, Callable] = {}
      GLOBAL_MAP.set_map(NNDCT_KEYS.NODE_CALLER_MAP, node2caller)
    node2caller[node.name] = torchop.caller
    op = TorchBaseOperation(schema_handler.op_name, torchop.name, schema=node.schema)
    # op.set_caller(torchop.caller)
    assert len(args) == len(schema_handler.get_arguments())
    if len(args) == 1:
        return op
    arg_name_convertor = {"self": "input"}
    for inp, arg in zip(args, schema_handler.get_arguments()):
      arg_name = schema_handler.arg_name(arg)
      if torchop.op_class_type == TorchOpClassType.TENSOR and arg_name == "self":
        continue
      if arg_name in ["layout", "memory_format", "pin_memory"]:
        continue
      config_name = arg_name_convertor.get(arg_name, arg_name)
      if convert_type_str(schema_handler.arg_type(arg)).replace("?", "") == "bool":
        inp = bool(inp) if inp is not None else inp
      if convert_type_str(schema_handler.arg_type(arg)).replace("?", "") == "str":
        inp = f"'{inp}'" if inp is not None else inp

      if arg_name == "device":
        inp = f"'{self._device_type}'"
      if arg_name == "dtype":
        inp = scalar_type_to_pytorch_type[inp] if inp is not None else inp
      op.set_config(config_name, inp)
    return op
    
  def custom_op(self, node, *args):
    node2caller = GLOBAL_MAP.get_ele(NNDCT_KEYS.NODE_CALLER_MAP)
    if node2caller is None:
      node2caller: Dict[str, Callable] = {}
      GLOBAL_MAP.set_map(NNDCT_KEYS.NODE_CALLER_MAP, node2caller)
    node2caller[node.name] = node.caller
    op = TorchCustomOperation(node.raw_kind, node.raw_kind)
    for i, arg in enumerate(args):
      op.set_config(str(i), arg)
    attrs = GLOBAL_MAP.get_ele(NNDCT_KEYS.CUSTOM_OP_ATTRS_MAP).get(node.raw_kind, None)
    if attrs:
      attr_vals = args[len(args)-len(attrs):]
      for name, val in zip(attrs, attr_vals):
        op.set_attr_by_name(name, val)
    return op
