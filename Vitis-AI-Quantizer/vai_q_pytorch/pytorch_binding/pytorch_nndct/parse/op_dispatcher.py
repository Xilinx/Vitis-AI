import numpy as np
import torch
import sys
from collections import namedtuple, defaultdict
import copy
from nndct_shared.base.key_names import FrameworkType
from nndct_shared.nndct_graph import Operation
from nndct_shared.nndct_graph import Tensor, Node
from nndct_shared.utils import tensor_util

from .utils import *
from .torch_op_def import *

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

  def __call__(self, graph_name, value):
    if isinstance(value.data, torch.Tensor):
      nndct_tensor = Tensor(
          name=get_full_name(graph_name, value.name),
          shape=value.shape,
          dtype=value.dtype,
          data=value.data.cpu().numpy())
    else:
      nndct_tensor = Tensor(
          name=get_full_name(graph_name, value.name), shape=value.shape, dtype=value.dtype)
    return nndct_tensor

class NodeConvertor(object):
  r"""convert raw_node to nndct_node"""

  def __call__(self, parser, raw_graph, raw_node):
    nndct_node = Node(
        name=get_full_name(raw_graph.name, raw_node.name),
        dtype=convert_dtype(raw_node.dtype),
        idx=raw_node.idx)

    blob_tensor_convertor = TensorConvertor()
    for op in raw_node.outputs:
      nndct_tensor = blob_tensor_convertor(raw_graph.name, op)
      nndct_tensor.node = nndct_node
      nndct_node.out_tensors.append(nndct_tensor)

    for ip in raw_node.flatten_inputs:
      if ip.name not in raw_graph.param_names(
      ) and parser.get_blob_tensor_by_name(ip.name):
        nndct_node.in_tensors.append(parser.get_blob_tensor_by_name(ip.name))
    return nndct_node


class OpCreator(object):

  def __call__(self, parser, raw_graph, raw_node):
    args = []
    params = {}
    param_tensor_convertor = TensorConvertor()
    if not raw_node.inputs:
      args.append(parser.get_nndct_value(raw_node.outputs[0]))
    else:
      infer_args = {}
      for ip in raw_node.flatten_inputs:
        if ip.name in raw_graph.param_names():
          param_tensor = param_tensor_convertor(raw_graph.name, ip)
          params[param_tensor.name] = param_tensor
          infer_args[ip.name] = param_tensor
        else:
          infer_args[ip.name] = parser.get_nndct_value(ip)

      for ip in raw_node.inputs:
        if isinstance(ip, list):
          args.append([infer_args[i.name] for i in ip])
        else:
          args.append(infer_args[ip.name])

    try:
      op = getattr(self, raw_node.kind, self.default)(*args)
    except NotImplementedError as e:
      print(f"aten_op '{raw_node.kind}' parse failed({str(e)})")
      sys.exit(1)

    self._setup_params(op, params)
    return op

  def _setup_params(self, nndct_op, params):

    def _find_param_tensor(nndct_param_name):
      for param_name in params:
        name = get_short_name(param_name)
        if nndct_param_name in name or \
          set(nndct_param_name.split("_")).issubset(name.split(".")[-1].split("_")):
          return params[param_name]
        
    if not hasattr(nndct_op, "ParamName"): 
      return 
    
    if nndct_op.type == NNDCT_OP.BASIC_LSTM or nndct_op.type == NNDCT_OP.BASIC_GRU:
      param = defaultdict(list)  
      # lstm_params = [name for name in params.keys()]
      while params.keys():
        for nndct_param in nndct_op.ParamName:
          nndct_param_name = nndct_param.value
          param_tensor = _find_param_tensor(nndct_param_name)
          if param_tensor:
            del params[param_tensor.name]
            param[nndct_param_name].append(param_tensor)
            nndct_op.set_param(nndct_param, param[nndct_param_name])
    else:
      for nndct_param in nndct_op.ParamName:
        nndct_param_name = nndct_param.value
        param_tensor = _find_param_tensor(nndct_param_name)
        if param_tensor:
          nndct_op.set_param(nndct_param, param_tensor)
          
  # torch function should always add 'input' config
  # nn.function, nn.Module and torch.Tensor can ignore 'input'
  
  def Param(self, *args):
    op = TorchBaseOperation(NNDCT_OP.INPUT, "input")
    input_name = f"args[{args[0].node.name.split('_')[-1]}]" 
    op.set_config("input", input_name)
    return op

  def _convolution(self, input, weight, bias, stride, padding, dilation,
                   transposed, output_padding, groups, benchmark, deterministic,
                   cudnn_enabled):
    # weight_size = weight.type().sizes()
    weight_size = weight.shape
    if transposed:
      weight_size[0], weight_size[1] = weight_size[1], weight_size[0]
      op = TorchConvTranspose2d()
      op.set_config("output_padding", list(output_padding))
      op.set_config('in_channels', weight_size[1])
      op.set_config('out_channels', weight_size[0] * groups)
    else:
      if weight_size[1] == 1:
        op = TorchConv2d(NNDCT_OP.DEPTHWISE_CONV2D)
      else:
        op = TorchConv2d(NNDCT_OP.CONV2D)
      op.set_config('in_channels', weight_size[1] * groups)
      op.set_config('out_channels', weight_size[0])

    if bias is not None:
      op.set_config('bias', True)
    else:
      op.set_config('bias', False)

    op.set_config('dilation', list(dilation))
    op.set_config('kernel_size', list(weight_size[2:]))
    op.set_config('stride', list(stride))
    # op.set_config('in_channels', weight_size[1] * groups)
    # op.set_config('out_channels', weight_size[0])
    op.set_config('groups', groups)
    op.set_config('padding', list(padding))
    return op

  def batch_norm(self, input, weight, bias, running_mean, running_var, training,
                 momentum, eps, cudnn_enabled):
    op = TorchBatchNorm(len(input.shape))
    weight_size = weight.shape
    if training:
      op.set_config('is_training', True)
    op.set_config('num_features', weight_size[0])
    op.set_config('eps', eps)
    op.set_config('momentum', momentum)
    op.set_attr(op.AttrName.SCALE, True)
    op.set_attr(op.AttrName.CENTER, True)

    return op

  @staticmethod
  def _max_pool2d(op, input, kernel_size, stride, padding, dilation, ceil_mode):
    op.set_config('kernel_size', list(kernel_size))
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
    op = TorchAdaptiveAvgPool()
    op.set_config('output_size', output_size)
    return op

  def addmm(self, bias, input, weight, beta, alpha):
    op = TorchLinear()
    weight_size = weight.shape

    if bias is None:
      op.set_config("bias", False)
    else:
      op.set_config("bias", True)

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

  def add(self, input, other, alpha):
    if input.is_complete_tensor() or other.is_complete_tensor():   
      op = TorchAdd()
    else:
      op = TorchBaseOperation(NNDCT_OP.SCALAR_ADD, "add")
      
    op.set_config('input', input)
    op.set_config('other', other)
    op.set_config('alpha', alpha)
    return op

  def size(self, input, dim):
    op = TorchSize(len(input.shape))
    op.set_config("dim", dim)
    return op

  def view(self, input, shape):
    op = TorchView(len(input.shape))
    op.set_config("size", shape)
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
    op = TorchCat(len(tensors[0].shape))
    op.set_config("dim", dim)
    op.set_config("tensors", tensors)
    return op

  def mean(self, input, dim=None, keepdim=False, dtype=None):
    op = TorchMean(len(input.shape))
    op.set_config("input", input)
    op.set_config("dim", dim)
    op.set_config("keepdim", keepdim)
    return op

  def relu6(self, input, inplace):
    op = TorchBaseOperation(NNDCT_OP.RELU6, "ReLU6")
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
    op = TorchTranspose(len(input.shape))
    op.set_config("input", input)
    op.set_config("dim0", dim0)
    op.set_config("dim1", dim1)
    op.get_config("dim0")
    op.get_config("dim1")

    return op

  def contiguous(self, *args):
    op = TorchContiguous()
    return op

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
                   align_corners=None):
    if mode == "'nearest'":
      op = TorchInterpolate(len(input.shape))
    else:
      op = TorchResizeLinear(len(input.shape))
      
    op.set_config("input", input)
    op.set_config("mode", mode)
    if size is not None:
        op.set_config("size", size)
    if align_corners is not None:
      op.set_config("align_corners", bool(align_corners))
    if scale_factor is not None: 
      op.set_config("scale_factor", scale_factor)
    return op

  def upsample_bilinear2d(self, input, tensor_list, align_corners):
    return self._interpolate(
        input,
        size=tensor_list,
        mode="'bilinear'",
        align_corners=align_corners)

  def upsample_nearest2d(self, input, tensor_list, *args):
    return self._interpolate(input, size=tensor_list)

  def NumToTensor(self, input, *args):
    op = TorchBaseOperation(NNDCT_OP.TENSOR, "tensor")
    op.set_config("data", input)
    op.set_config("dtype", "torch.float32")
    op.set_config("device", "'cuda'")
    return op

  def Constant(self, tensor, *args):
    op = TorchConst()
    # tensor_util.convert_blob_tensor_format(tensor, FrameworkType.NNDCT, FrameworkType.TORCH)
    op.set_config('data', tensor.data.tolist())
    op.set_config('dtype', tensor.dtype)
    op.set_config('device', "'cuda'")
    # tensor_util.convert_blob_tensor_format(tensor, FrameworkType.TORCH, FrameworkType.NNDCT)

    return op

  def mul(self, input, other):
    if input.is_complete_tensor() or other.is_complete_tensor():   
      op = TorchMul()
    else:
      op = TorchBaseOperation(NNDCT_OP.SCALAR_MUL, "mul")
      
    op.set_config("input", input)
    op.set_config("other", other)
    return op

  def to(self, input, *args):
    op = TorchCast()
    if isinstance(args[0], str):
      op.set_config('dtype', scalar_type_to_pytorch_type[args[1]])
    else:
      op.set_config('dtype', scalar_type_to_pytorch_type[args[0]])

    return op

  def floor(self, input):
    op = TorchFloor()
    op.set_config("input", input)
    return op

  def Int(self, *args):
    op = TorchBaseOperation(NNDCT_OP.INT, "int")
    op.set_config("input", args[0])
    return op

  def permute(self, input, dims):
    op = TorchPermute(len(input.shape))
    op.set_config("dims", dims)
    # op.get_attr(op.AttrName.ORDER)
    return op

  def div(self, input, other):
    op = TorchDiv()
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

  def strided_slice(self, input, dim, start, end, step):
    op = TorchSlice(len(input.shape))
    begin = [0] * len(input.shape)
    last = [NNDCT_CONSTANT.INT_MAX] * len(input.shape)
    stride = [1] * len(input.shape)
    for i, pos in enumerate(dim):
      begin[pos] = start[i]
      if end[i] < last[pos]:
        last[pos] = end[i]
      stride[pos] = step[i]
    op.set_config("input", input)
    op.set_config("start", begin)
    op.set_config("end", last)
    op.set_config("step", stride)
    return op

  def sub(self, input, other, alpha):
    op = TorchSub()
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
    op = TorchExpand()
    op.set_config('size', size)
    return op

  def t(self, input, *args):
    return self.transpose(input, 0, 1)

  def empty(self, *args):
    op = TorchEmpty()
    op.set_config('size', args[0])
    op.set_config('dtype', scalar_type_to_pytorch_type[args[1]])
    op.set_config('device', "'cuda'")
    return op

  # def ListConstruct(self, *args):
  #   op = TorchList()
  #   op.set_config('input', args)
  #   return op

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
    return op

  def gru(self, *args):
    if isinstance(args[3], list):
      raise NotImplementedError('Unimplement packed gri')
    else:
      return self._gru_full(*args)
  
  def _gru_full(self, input, hidden, weight_v, has_biases,
                 num_layers, dropout, train, bidirectional, batch_first):
    op = TorchGru()
    op.set_config('bias', bool(has_biases))
    op.set_config('bidirectional', bool(bidirectional))
    op.set_config('input_size', input.shape[-1])
    op.set_config('hidden_size', hidden.shape[-1])
    op.set_config('batch_first', bool(batch_first))
    op.set_config('num_layers', num_layers)
    op.set_config('dropout', dropout)
    return op
  
  def zeros(self, sizes, dtype, layout, device, pin_memory=False):
    op = TorchZeros()
    op.set_config('size', sizes)
    op.set_config('dtype', scalar_type_to_pytorch_type[dtype])
    op.set_config('device', "'cuda'")
    return op

  def constant_pad_nd(self, input, pad, value):
    op = TorchPad()
    op.set_config('input', input)
    op.set_config('pad', pad)
    op.set_config('value', value)
    op.set_config('mode', "'constant'")
    return op

  def matmul(self, input, other, *args):
    if "weight" in other.name:
      op = TorchLinear()
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
    return op

  def clamp_min(self, input, min):
    op = TorchClamp()
    op.set_config("input", input)
    op.set_config("min", min)
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
      op.set_config("device", "'cuda'")
    elif len(args) == 6:
      # aten::arange(Scalar start, Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
      op.set_config("start", args[0])
      op.set_config("end", args[2])
      op.set_config("dtype", scalar_type_to_pytorch_type[4])
      op.set_config("device", "'cuda'")
    elif len(args) == 7:
      # aten::arange(Scalar start, Scalar end, Scalar step, ScalarType dtype, Layout, Device, bool pin_memory)
      op.set_config("start", args[0])
      op.set_config("end", args[1])
      op.set_config("step", args[2])
      op.set_config("dtype", scalar_type_to_pytorch_type[args[3]])
      op.set_config("device", "'cuda'")
    else:
      raise NotImplementedError("Unknown aten::arange signature taking " +
                                str(len(args)) + " arguments.")
    return op

  def slice_tensor_inplace_copy(self, input, src, non_blocking, dim, index):
    op = TorchBaseOperation(NNDCT_OP.SLICE_TENSOR_INPLACE_COPY, NNDCT_OP.SLICE_TENSOR_INPLACE_COPY)
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
    op.set_config("p", "'fro'")
    return op
  
  def expand_as(self, input, other):
    op = TorchBaseOperation(NNDCT_OP.EXPAND_AS, "expand_as")
    op.set_config("other", other)
    return op

  def max(self, input, dim, keepdim):
    op = TorchBaseOperation(NNDCT_OP.MAX, "max")
    op.set_config("input", input)
    op.set_config("dim", dim)
    op.set_config("keepdim", bool(keepdim))
    return op
  
  def squeeze(self, input):
    op = TorchBaseOperation(NNDCT_OP.SQUEEZE, "squeeze")
    op.set_config("input", input)
    return op
  
  def eq(self, input, other):
    op = TorchBaseOperation(NNDCT_OP.EQUAL, "eq")
    op.set_config("input", input)
    op.set_config("other", other)
    return op

  def index(self, input, index):
    op = TorchBaseOperation(NNDCT_OP.INDEX, NNDCT_OP.INDEX)
    op.set_config("input", input)
    op.set_config("index", index)
    return op
  
  def index_put_(self, input, indices, value, accumulate):
    op = TorchBaseOperation(NNDCT_OP.INDEX_INPUT_INPLACE, NNDCT_OP.INDEX_INPUT_INPLACE)
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

  def feature_dropout(self, *args):
    return self.dropout(*args)
  
  def default(self, *args):
    raise NotImplementedError("unsupported")
