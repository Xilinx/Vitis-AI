from enum import auto, unique

import pytorch_nndct.utils as utils
from nndct_shared.base import NNDCT_CONSTANT, NNDCT_OP
from nndct_shared.nndct_graph import Operation
from nndct_shared.nndct_graph import operator_definition as base_op


def _transformed_axis(src: str, dst: str, ndim: int, dim: int) -> int:
  """NCHW -> NHWC/ NHWC ->NCHW"""
  if ndim != 4:
    return dim
  if src == dst:
    return dim
  if src == "NCHW" and dst == "NHWC":
    return dim + [0, 2, -1, -1][dim]
  elif src == "NHWC" and dst == "NCHW":
    return dim + [0, 1, 1, -2][dim]


class TorchFlatten(base_op.Flatten):

  def __init__(self, *args, **kwargs):
    super(TorchFlatten, self).__init__(NNDCT_OP.FLATTEN, *args, **kwargs)
    utils.op_register(NNDCT_OP.FLATTEN, 'flatten')

  @property
  def start_dim(self):
    return self._attr_value_mem[self.AttrName.START_DIM][0]

  @start_dim.setter
  def start_dim(self, value):
    self._attr_value_mem[self.AttrName.START_DIM][:] = [value]

  @property
  def end_dim(self):
    return self._attr_value_mem[self.AttrName.END_DIM][0]

  @end_dim.setter
  def end_dim(self, value):
    self._attr_value_mem[self.AttrName.END_DIM][:] = [value]


class TorchAdd(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchAdd, self).__init__(NNDCT_OP.ADD, *args, **kwargs)
    utils.op_register(NNDCT_OP.ADD, 'add')


class TorchReLU(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchReLU, self).__init__(NNDCT_OP.RELU, *args, **kwargs)
    utils.op_register(NNDCT_OP.RELU, 'ReLU')


class TorchLeakyReLU(base_op.LeakyReLU):

  def __init__(self):
    super().__init__()
    utils.op_register(NNDCT_OP.LEAKY_RELU, 'LeakyReLU')
    self._negative_slope = 0.01
    
  @property
  def negative_slope(self):
    return self._negative_slope

  @negative_slope.setter
  def negative_slope(self, value):
    self._negative_slope = value
    self.set_attr(self.AttrName.ALPHA, 0.1015625)


class TorchTanh(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchTanh, self).__init__(NNDCT_OP.TANH, *args, **kwargs)
    utils.op_register(NNDCT_OP.TANH, 'Tanh')


class TorchHardTanh(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchHardTanh, self).__init__(NNDCT_OP.HARDTANH, *args, **kwargs)
    utils.op_register(NNDCT_OP.HARDTANH, 'Hardtanh')


# class TorchInput(Operation):

#   def __init__(self, *args, **kwargs):
#     super(TorchInput, self).__init__(NNDCT_OP.INPUT, *args, **kwargs)

class TorchLinear(base_op.Dense):

  @unique
  class ParamName(base_op.AutoName):
    WEIGHTS = "weight"
    BIAS = auto()
    
  def __init__(self, *args, **kwargs):
    super(TorchLinear, self).__init__(NNDCT_OP.DENSE, *args, **kwargs)
    utils.op_register(NNDCT_OP.DENSE, 'Linear')

  @property
  def bias(self):
    return self._attr_value_mem[self.AttrName.BIAS_TERM][0]

  @bias.setter
  def bias(self, value):
    self._attr_value_mem[self.AttrName.BIAS_TERM][:] = [bool(value)]

  @property
  def in_features(self):
    return self._attr_value_mem[self.AttrName.IN_DIM][0]

  @in_features.setter
  def in_features(self, value):
    self._attr_value_mem[self.AttrName.IN_DIM][:] = [value]

  @property
  def out_features(self):
    return self._attr_value_mem[self.AttrName.OUT_DIM][0]

  @out_features.setter
  def out_features(self, value):
    self._attr_value_mem[self.AttrName.OUT_DIM][:] = [value]


class TorchBatchNorm(base_op.BatchNorm):

  @unique
  class ParamName(base_op.AutoName):
    GAMMA = "weight"
    BETA = "bias"
    MOVING_MEAN = "mean"
    MOVING_VAR = "var"

  def __init__(self, dim):
    if dim == 2 or dim == 3:
      nndct_op_type = NNDCT_OP.BATCH_NORM1D
      torch_op_type = "BatchNorm1d"
    elif dim == 4:
      nndct_op_type = NNDCT_OP.BATCH_NORM
      torch_op_type = "BatchNorm2d"
    else:
      nndct_op_type = NNDCT_OP.BATCH_NORM3D
      torch_op_type = "BatchNorm3d"
      
    super().__init__(nndct_op_type)
    utils.op_register(nndct_op_type, torch_op_type)

  @property
  def eps(self):
    return self._attr_value_mem[self.AttrName.EPSILON][0]

  @eps.setter
  def eps(self, value):
    self._attr_value_mem[self.AttrName.EPSILON][:] = [value]

  @property
  def num_features(self):
    return self._attr_value_mem[self.AttrName.OUT_DIM][0]

  @num_features.setter
  def num_features(self, value):
    self._attr_value_mem[self.AttrName.OUT_DIM][:] = [value]


class _TorchConv2d(base_op.Conv2d):

  @unique
  class ParamName(base_op.AutoName):
    WEIGHTS = "weight"
    BIAS = auto()
    
  def __init__(self, op_type, *args, **kwargs):
    super(_TorchConv2d, self).__init__(op_type, *args, **kwargs)

  @property
  def kernel_size(self):
    return self._attr_value_mem[self.AttrName.KERNEL][::-1]

  @kernel_size.setter
  def kernel_size(self, value):
    self._attr_value_mem[self.AttrName.KERNEL][:] = value[::-1]

  @property
  def dilation(self):
    return self._attr_value_mem[self.AttrName.DILATION][::-1]

  @dilation.setter
  def dilation(self, value):
    self._attr_value_mem[self.AttrName.DILATION][:] = value[::-1]

  @property
  def padding(self):
    return [
        self._attr_value_mem[self.AttrName.PAD][2],
        self._attr_value_mem[self.AttrName.PAD][0]
    ]

  @padding.setter
  def padding(self, value):
    self._attr_value_mem[self.AttrName.PAD_MODE][:] = [0]
    self._attr_value_mem[self.AttrName.PAD][:] = [
        value[1], value[1], value[0], value[0]
    ]

  @property
  def stride(self):
    return self._attr_value_mem[self.AttrName.STRIDE][::-1]

  @stride.setter
  def stride(self, value):
    self._attr_value_mem[self.AttrName.STRIDE][:] = value[::-1]

  @property
  def in_channels(self):
    return self._attr_value_mem[self.AttrName.IN_DIM][0]

  @in_channels.setter
  def in_channels(self, value):
    self._attr_value_mem[self.AttrName.IN_DIM][:] = [value]

  @property
  def out_channels(self):
    return self._attr_value_mem[self.AttrName.OUT_DIM][0]

  @out_channels.setter
  def out_channels(self, value):
    self._attr_value_mem[self.AttrName.OUT_DIM][:] = [value]

  @property
  def groups(self):
    return self._attr_value_mem[self.AttrName.GROUP][0]

  @groups.setter
  def groups(self, value):
    self._attr_value_mem[self.AttrName.GROUP][:] = [value]

  @property
  def bias(self):
    return self._attr_value_mem[self.AttrName.BIAS_TERM][0]

  @bias.setter
  def bias(self, value):
    self._attr_value_mem[self.AttrName.BIAS_TERM][:] = [bool(value)]


class TorchConv2d(_TorchConv2d):

  def __init__(self, nndct_op_type, *args, **kwargs):
    super(TorchConv2d, self).__init__(nndct_op_type, *args, **kwargs)
    utils.op_register(nndct_op_type, "Conv2d")


class TorchConvTranspose2d(_TorchConv2d):

  def __init__(self, *args, **kwargs):
    super(TorchConvTranspose2d, self).__init__(NNDCT_OP.CONVTRANSPOSE2D, *args,
                                               **kwargs)
    utils.op_register(NNDCT_OP.CONVTRANSPOSE2D, "ConvTranspose2d")


class TorchMaxPool(base_op.MaxPool):

  def __init__(self, *args, **kwargs):
    super(TorchMaxPool, self).__init__(NNDCT_OP.MAX_POOL, *args, **kwargs)
    utils.op_register(NNDCT_OP.MAX_POOL, "MaxPool2d")

  @property
  def kernel_size(self):
    return self._attr_value_mem[self.AttrName.KERNEL][::-1]

  @kernel_size.setter
  def kernel_size(self, value):
    self._attr_value_mem[self.AttrName.KERNEL][:] = value[::-1]

  @property
  def ceil_mode(self):
    return bool(self._attr_value_mem[self.AttrName.PAD_MODE][0])

  @ceil_mode.setter
  def ceil_mode(self, mode):
    self._attr_value_mem[self.AttrName.PAD_MODE][:] = [int(mode)]

  @property
  def padding(self):
    return [
        self._attr_value_mem[self.AttrName.PAD][2],
        self._attr_value_mem[self.AttrName.PAD][0]
    ]

  @padding.setter
  def padding(self, value):
    self._attr_value_mem[self.AttrName.PAD][:] = [
        value[1], value[1], value[0], value[0]
    ]

  @property
  def stride(self):
    return self._attr_value_mem[self.AttrName.STRIDE][::-1]

  @stride.setter
  def stride(self, value):
    self._attr_value_mem[self.AttrName.STRIDE][:] = value[::-1]


class TorchAvgPool(base_op.AvgPool):

  def __init__(self, *args, **kwargs):
    super(TorchAvgPool, self).__init__(NNDCT_OP.AVG_POOL, *args, **kwargs)
    utils.op_register(NNDCT_OP.AVG_POOL, "AvgPool2d")
    # set default attr
    self.set_attr(self.AttrName.GLOBAL, False)

  @property
  def kernel_size(self):
    return self._attr_value_mem[self.AttrName.KERNEL][::-1]

  @kernel_size.setter
  def kernel_size(self, value):
    self._attr_value_mem[self.AttrName.KERNEL][:] = value[::-1]

  @property
  def ceil_mode(self):
    return bool(self._attr_value_mem[self.AttrName.PAD_MODE][0])

  @ceil_mode.setter
  def ceil_mode(self, mode):
    self._attr_value_mem[self.AttrName.PAD_MODE][:] = [int(mode)]

  @property
  def padding(self):
    return [
        self._attr_value_mem[self.AttrName.PAD][2],
        self._attr_value_mem[self.AttrName.PAD][0]
    ]

  @padding.setter
  def padding(self, value):
    self._attr_value_mem[self.AttrName.PAD][:] = [
        value[1], value[1], value[0], value[0]
    ]

  @property
  def stride(self):
    return self._attr_value_mem[self.AttrName.STRIDE][::-1]

  @stride.setter
  def stride(self, value):
    self._attr_value_mem[self.AttrName.STRIDE][:] = value[::-1]


class TorchAdaptiveAvgPool(base_op.AvgPool):

  def __init__(self, *args, **kwargs):
    super(TorchAdaptiveAvgPool, self).__init__(NNDCT_OP.ADAPTIVEAVGPOOL2D,
                                               *args, **kwargs)
    utils.op_register(NNDCT_OP.ADAPTIVEAVGPOOL2D, 'AdaptiveAvgPool2d')
    # set default value
    self.set_attr(self.AttrName.KERNEL, [1, 1])
    self.set_attr(self.AttrName.STRIDE, [1, 1])
    self.set_attr(self.AttrName.PAD_MODE, 0)

  def __setattr__(self, key, value):
    if key == "output_size":
      if value != [1, 1] and value != 1:
        self._attr_value_mem[self.AttrName.GLOBAL][:] = [False]
      else:
        self._attr_value_mem[self.AttrName.GLOBAL][:] = [True]
    self.__dict__[key] = value


class TorchSize(base_op.Shape):

  def __init__(self, input_ndim, *args, **kwargs):
    super(TorchSize, self).__init__(NNDCT_OP.SHAPE, *args, **kwargs)
    utils.op_register(NNDCT_OP.SHAPE, 'size')
    self._input_ndim = input_ndim

  @property
  def dim(self):
    return _transformed_axis(
        src="NHWC",
        dst="NCHW",
        ndim=self._input_ndim,
        dim=self._attr_value_mem[self.AttrName.AXIS][0])

  @dim.setter
  def dim(self, value):
    self._attr_value_mem[self.AttrName.AXIS][:] = [
        _transformed_axis(
            src="NCHW", dst="NHWC", ndim=self._input_ndim, dim=value)
    ]


class TorchCat(base_op.Concat):

  def __init__(self, input_ndim, *args, **kwargs):
    super(TorchCat, self).__init__(NNDCT_OP.CONCAT, *args, **kwargs)
    utils.op_register(NNDCT_OP.CONCAT, 'cat')
    self._input_ndim = input_ndim

  @property
  def dim(self):
    return _transformed_axis(
        src="NHWC",
        dst="NCHW",
        ndim=self._input_ndim,
        dim=self._attr_value_mem[self.AttrName.AXIS][0])

  @dim.setter
  def dim(self, value):
    self._attr_value_mem[self.AttrName.AXIS][:] = [
        _transformed_axis(
            src="NCHW", dst="NHWC", ndim=self._input_ndim, dim=value)
    ]


class TorchView(base_op.Reshape):

  def __init__(self, input_ndim, *args, **kwargs):
    super(TorchView, self).__init__(NNDCT_OP.RESHAPE, *args, **kwargs)
    utils.op_register(NNDCT_OP.RESHAPE, 'view')
    self._input_ndim = input_ndim

  @property
  def size(self):
    if len(self._attr_value_mem[self.AttrName.SHAPE]) == 1:
      return self._attr_value_mem[self.AttrName.SHAPE][0]
    else:
      return self._attr_value_mem[self.AttrName.SHAPE][:]

  @size.setter
  def size(self, value):
    if isinstance(value, (tuple, list)):
      value = list(value)
    else:
      value = [value]

    if self._input_ndim != len(value):
      self._attr_value_mem[self.AttrName.SHAPE][:] = value[:]
    else:
      raise RuntimeError(
          f"the layout of activation of {self.type} is ambiguous")


class TorchDropout(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchDropout, self).__init__(NNDCT_OP.DROPOUT, *args, **kwargs)
    utils.op_register(NNDCT_OP.DROPOUT, 'Dropout')


class TorchMean(base_op.Mean):
 
  def __init__(self, input_ndim, *args, **kwargs):
    super(TorchMean, self).__init__(NNDCT_OP.MEAN, *args, **kwargs)
    utils.op_register(NNDCT_OP.MEAN, 'mean')
    self._input_ndim = input_ndim

  @property
  def dim(self):
    dims = []  
    for dim in self.get_attr(self.AttrName.DIMS):
      dims.append(
          _transformed_axis(
              src="NHWC", dst="NCHW", ndim=self._input_ndim, dim=dim))
    return tuple(dims)

  @dim.setter
  def dim(self, value):
    if isinstance(value, (tuple, list)):
      value = list(value)
    else:
      value = [value]
    dims = []
    for dim in value:
      dims.append(
          _transformed_axis(
              src="NCHW", dst="NHWC", ndim=self._input_ndim, dim=dim))
    
    self.set_attr(self.AttrName.DIMS, [dims[:]])

  @property
  def keepdim(self):
    return self._attr_value_mem[self.AttrName.KEEP_DIMS][0]

  @keepdim.setter
  def keepdim(self, value):
    self._attr_value_mem[self.AttrName.KEEP_DIMS][:] = [bool(value)]


class TorchPermute(base_op.Permute):

  def __init__(self, input_ndim, *args, **kwargs):
    super(TorchPermute, self).__init__(NNDCT_OP.PERMUTE, *args, **kwargs)
    utils.op_register(NNDCT_OP.PERMUTE, 'permute')
    self._input_ndim = input_ndim

  @property
  def dims(self):
    dims = []
    for dim in self._attr_value_mem[self.AttrName.ORDER]:
      dims.append(
          _transformed_axis(
              src="NHWC", dst="NCHW", ndim=self._input_ndim, dim=dim))
    return dims

  @dims.setter
  def dims(self, value):
    dims = []
    for dim in value:
      dims.append(
          _transformed_axis(
              src="NCHW", dst="NHWC", ndim=self._input_ndim, dim=dim))
    self._attr_value_mem[self.AttrName.ORDER][:] = dims[:]


class TorchTranspose(base_op.Permute):

  def __init__(self, input_ndim, *args, **kwargs):
    super(TorchTranspose, self).__init__(NNDCT_OP.TRANSPOSE, *args, **kwargs)
    utils.op_register(NNDCT_OP.TRANSPOSE, 'transpose')
    self._input_ndim = input_ndim
    self._dim0 = None
    self._dim1 = None
    self._attr_value_mem[self.AttrName.ORDER][:] = list(range(input_ndim))

  def _exchange_dims_in_attrs(self):
    self._attr_value_mem[self.AttrName.ORDER][_transformed_axis(
        src="NCHW", dst="NHWC", ndim=self._input_ndim,
        dim=self._dim0)] = _transformed_axis(
            src="NCHW", dst="NHWC", ndim=self._input_ndim, dim=self._dim1)
    self._attr_value_mem[self.AttrName.ORDER][_transformed_axis(
        src="NCHW", dst="NHWC", ndim=self._input_ndim,
        dim=self._dim1)] = _transformed_axis(
            src="NCHW", dst="NHWC", ndim=self._input_ndim, dim=self._dim0)

  @property
  def dim0(self):
    return self._attr_value_mem[self.AttrName.ORDER][_transformed_axis(
        src="NHWC", dst="NCHW", ndim=self._input_ndim, dim=self._dim1)]

  @dim0.setter
  def dim0(self, value):
    self._dim0 = value
    if self._dim1 is not None:
      self._exchange_dims_in_attrs()

  @property
  def dim1(self):
    return self._attr_value_mem[self.AttrName.ORDER][_transformed_axis(
        src="NHWC", dst="NCHW", ndim=self._input_ndim, dim=self._dim0)]

  @dim1.setter
  def dim1(self, value):
    self._dim1 = value
    if self._dim0 is not None:
      self._exchange_dims_in_attrs()


class TorchContiguous(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchContiguous, self).__init__(NNDCT_OP.CONTIGUOUS, *args, **kwargs)
    utils.op_register(NNDCT_OP.CONTIGUOUS, 'contiguous')


class TorchChunk(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchChunk, self).__init__(NNDCT_OP.CHUNK, *args, **kwargs)
    utils.op_register(NNDCT_OP.CHUNK, 'chunk')


class TorchInterpolate(base_op.Resize):
  def __init__(self, input_ndim):
    super().__init__()
    utils.op_register(NNDCT_OP.RESIZE, 'interpolate')
    self._scale_factor_bc = [1.0, 1.0]
    if input_ndim != 4:
        raise RuntimeError("Only support 2D unsampling.")
 
  @property
  def size(self):
    size = self.get_attr(self.AttrName.SIZE)[:]
    if size[0] == 0 and size[1] == 0:
      return None
    else:
      return size

  @size.setter
  def size(self, size):
    self.set_attr(self.AttrName.SIZE, size[:])
    
  @property
  def scale_factor(self):
    scale = self._scale_factor_bc + self.get_attr(self.AttrName.SCALE)[::-1]
    if self.size[0] == 0 and self.size[0] == 0:
      return scale
    else:
      return None
  
  @scale_factor.setter
  def scale_factor(self, factor):
    if isinstance(factor, float):
        self._scale_factor_bc = 2 * [factor]
    else:
        self._scale_factor_bc = factor[:2]
    self.set_attr(self.AttrName.SCALE, factor[2::-1])

  # @property
  # def align_corners(self):
  #   return self.get_attr(self.AttrName.ALIGN_CORNERS)

  # @align_corners.setter
  # def align_corners(self, value):
  #   self.set_attr(self.AttrName.ALIGN_CORNERS, value)
  
  @property
  def mode(self):
    mode = self.get_attr(self.AttrName.MODE)
    return "'nearest'" if mode == 0 else "'bilinear'"
    
  @mode.setter
  def mode(self, mode):
    if mode not in ["'nearest'", "'bilinear'"]:
      raise RuntimeError(f"Don't support {mode} mode in upsampling.")
    mode = 0 if mode == "'nearest'" else 3
    self.set_attr(self.AttrName.MODE, mode)

    
class TorchResizeLinear(TorchInterpolate):
  def __init__(self, input_ndim):
    super().__init__(input_ndim)
  
  @property
  def align_corners(self):
    return self.get_attr(self.AttrName.ALIGN_CORNERS)

  @align_corners.setter
  def align_corners(self, value):
    self.set_attr(self.AttrName.ALIGN_CORNERS, bool(value))
    self.set_attr(self.AttrName.HALF_PIXEL_CENTERS, not(bool(value)))


class TorchConst(base_op.Constant):
  def __init__(self):
    super().__init__(NNDCT_OP.CONST)
    utils.op_register(NNDCT_OP.CONST, 'tensor')
  
  @property
  def data(self):
    return self.get_attr(self.AttrName.DATA)
  
  @data.setter
  def data(self, data):
    self.set_attr(self.AttrName.DATA, data)
    
# class TorchTensor(Operation):

#   def __init__(self):
#     super().__init__(NNDCT_OP.TENSOR)
#     utils.op_register(NNDCT_OP.TENSOR, 'tensor')

#   # @property
#   # def data(self):
#   #   return self.get_attr(self.AttrName.DATA)
  
#   # @data.setter
#   # def data(self, data):
#   #   self.set_attr(self.AttrName.DATA, data)
    
    
class TorchMul(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchMul, self).__init__(NNDCT_OP.MULTIPLY, *args, **kwargs)
    utils.op_register(NNDCT_OP.MULTIPLY, 'mul')


class TorchCast(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchCast, self).__init__(NNDCT_OP.CAST, *args, **kwargs)
    utils.op_register(NNDCT_OP.CAST, 'to')


class TorchFloor(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchFloor, self).__init__(NNDCT_OP.FLOOR, *args, **kwargs)
    utils.op_register(NNDCT_OP.FLOOR, 'floor')


# class TorchInt(Operation):

#   def __init__(self, *args, **kwargs):
#     super(TorchInt, self).__init__(NNDCT_OP.INT, *args, **kwargs)


class TorchDiv(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchDiv, self).__init__(NNDCT_OP.DEVIDE, *args, **kwargs)
    utils.op_register(NNDCT_OP.DEVIDE, 'div')


class TorchSoftmax(base_op.Softmax):

  def __init__(self, input_ndim, *args, **kwargs):
    super().__init__()
    utils.op_register(NNDCT_OP.SOFTMAX, 'Softmax')
    self._input_ndim = input_ndim

  @property
  def dim(self):
    return _transformed_axis(
        src="NHWC",
        dst="NCHW",
        ndim=self._input_ndim,
        dim=self._attr_value_mem[self.AttrName.AXIS][0])

  @dim.setter
  def dim(self, value):
    self._attr_value_mem[self.AttrName.AXIS][:] = [
        _transformed_axis(
            src="NCHW", dst="NHWC", ndim=self._input_ndim, dim=value)
    ]


class TorchExp(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchExp, self).__init__(NNDCT_OP.EXP, *args, **kwargs)
    utils.op_register(NNDCT_OP.EXP, 'exp')


class TorchDetach(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchDetach, self).__init__(NNDCT_OP.DETACH, *args, **kwargs)
    utils.op_register(NNDCT_OP.DETACH, 'detach')


class TorchSub(base_op.Sub):

  def __init__(self):
    super().__init__(NNDCT_OP.SUB)
    utils.op_register(NNDCT_OP.SUB, 'sub')

  @property
  def input(self):
    return self.get_attr(self.AttrName.INPUT)

  @input.setter
  def input(self, input):
    self.set_attr(self.AttrName.INPUT, input)

  @property
  def other(self):
    return self.get_attr(self.AttrName.OTHER)

  @other.setter
  def other(self, other):
    self.set_attr(self.AttrName.OTHER, other)


class TorchRsub(base_op.Sub):

  def __init__(self):
    super().__init__(NNDCT_OP.RSUB)
    utils.op_register(NNDCT_OP.RSUB, 'sub')

  @property
  def input(self):
    return self.get_attr(self.AttrName.INPUT)

  @input.setter
  def input(self, input):
    self.set_attr(self.AttrName.INPUT, input)

  @property
  def other(self):
    return self.get_attr(self.AttrName.OTHER)

  @other.setter
  def other(self, other):
    self.set_attr(self.AttrName.OTHER, other)


class TorchSelect(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchSelect, self).__init__(NNDCT_OP.SELECT, *args, **kwargs)
    utils.op_register(NNDCT_OP.SELECT, 'select')


class TorchSigmoid(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchSigmoid, self).__init__(NNDCT_OP.SIGMOID, *args, **kwargs)
    utils.op_register(NNDCT_OP.SIGMOID, 'Sigmoid')


class TorchRepeat(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchRepeat, self).__init__(NNDCT_OP.REPEAT, *args, **kwargs)
    utils.op_register(NNDCT_OP.REPEAT, 'repeat')


class TorchInplaceCopy(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchInplaceCopy, self).__init__(NNDCT_OP.INPLACE_COPY, *args,
                                           **kwargs)
    utils.op_register(NNDCT_OP.INPLACE_COPY, 'copy_')


class TorchExpand(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchExpand, self).__init__(NNDCT_OP.EXPAND, *args, **kwargs)
    utils.op_register(NNDCT_OP.EXPAND, 'expand')


class TorchEmpty(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchEmpty, self).__init__(NNDCT_OP.EMPTY, *args, **kwargs)
    utils.op_register(NNDCT_OP.EMPTY, 'empty')


class TorchUnsqueeze(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchUnsqueeze, self).__init__(NNDCT_OP.UNSQUEEZE, *args, **kwargs)
    utils.op_register(NNDCT_OP.UNSQUEEZE, 'unsqueeze')


class TorchList(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchList, self).__init__(NNDCT_OP.LIST, *args, **kwargs)


class TorchLstm(base_op.Lstm):

  def __init__(self, *args, **kwargs):
    super(TorchLstm, self).__init__(NNDCT_OP.BASIC_LSTM, *args, **kwargs)
    utils.op_register(NNDCT_OP.BASIC_LSTM, 'LSTM')

  @property
  def input_size(self):
    return self._attr_value_mem[self.AttrName.INPUT_SIZE][0]

  @input_size.setter
  def input_size(self, value):
    self._attr_value_mem[self.AttrName.INPUT_SIZE][:] = [value]

  @property
  def hidden_size(self):
    return self._attr_value_mem[self.AttrName.HIDDEN_SIZE][0]

  @hidden_size.setter
  def hidden_size(self, value):
    self._attr_value_mem[self.AttrName.HIDDEN_SIZE][:] = [value]

  @property
  def bidirectional(self):
    return self._attr_value_mem[self.AttrName.BIDIRECTIONAL][0]

  @bidirectional.setter
  def bidirectional(self, value):
    self._attr_value_mem[self.AttrName.BIDIRECTIONAL][:] = [bool(value)]

  @property
  def num_layers(self):
    return self._attr_value_mem[self.AttrName.NUM_LAYERS][0]

  @num_layers.setter
  def num_layers(self, value):
    self._attr_value_mem[self.AttrName.NUM_LAYERS][:] = [value]

  @property
  def batch_first(self):
    return self._attr_value_mem[self.AttrName.BATCH_FIRST][0]

  @batch_first.setter
  def batch_first(self, value):
    self._attr_value_mem[self.AttrName.BATCH_FIRST][:] = [bool(value)]


class TorchGru(base_op.Gru):
  def __init__(self, *args, **kwargs):
    super(TorchGru, self).__init__(NNDCT_OP.BASIC_GRU, *args, **kwargs)
    utils.op_register(NNDCT_OP.BASIC_GRU, 'GRU')
  
  @property
  def input_size(self):
    return self._attr_value_mem[self.AttrName.INPUT_SIZE][0]
  
  @input_size.setter
  def input_size(self, value):
    self._attr_value_mem[self.AttrName.INPUT_SIZE][:] = [value]
  
  @property
  def hidden_size(self):
    return self._attr_value_mem[self.AttrName.HIDDEN_SIZE][0]
  
  @hidden_size.setter
  def hidden_size(self, value):
    self._attr_value_mem[self.AttrName.HIDDEN_SIZE][:] = [value]
    
  @property
  def bidirectional(self):
    return self._attr_value_mem[self.AttrName.BIDIRECTIONAL][0]
  
  @bidirectional.setter
  def bidirectional(self, value):
    self._attr_value_mem[self.AttrName.BIDIRECTIONAL][:] = [bool(value)]
  
  @property
  def num_layers(self):
    return self._attr_value_mem[self.AttrName.NUM_LAYERS][0]
  
  @num_layers.setter
  def num_layers(self, value):
    self._attr_value_mem[self.AttrName.NUM_LAYERS][:] = [value]
  
  @property
  def batch_first(self):
    return self._attr_value_mem[self.AttrName.BATCH_FIRST][0]
  
  @batch_first.setter
  def batch_first(self, value):
    self._attr_value_mem[self.AttrName.BATCH_FIRST][:] = [bool(value)]  

  
class TorchSplit(Operation):

  def __init__(self, input_ndim, *args, **kwargs):
    super(TorchSplit, self).__init__(NNDCT_OP.SPLIT, *args, **kwargs)
    utils.op_register(NNDCT_OP.SPLIT, 'split')


class TorchZeros(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchZeros, self).__init__(NNDCT_OP.ZEROS, *args, **kwargs)
    utils.op_register(NNDCT_OP.ZEROS, 'zeros')


class TorchPad(base_op.Pad):

  def __init__(self):
    super().__init__()
    utils.op_register(self.type, "pad")

  @property
  def pad(self):
    return self.get_attr(self.AttrName.PAD_WITH)

  @pad.setter
  def pad(self, value):
    if len(value) != 4:
      raise RuntimeError("only support 2D pad")
    self.set_attr(self.AttrName.PAD_WITH, value)

  @property
  def mode(self):
    mode = self.get_attr(self.AttrName.MODE)
    return "'constant'" if mode == 0 else "'reflect'"

  @mode.setter
  def mode(self, mode):
    if mode not in ["'constant'", "'reflect'"]:
      raise RuntimeError(f"mode `{mode}` not supported in pad.")
    mode = 0 if mode == "'constant'" else 1
    self.set_attr(self.AttrName.MODE, mode)

  @property
  def value(self):
    return self.get_attr(self.AttrName.CONSTANT_VALUES)[0]

  @value.setter
  def value(self, constant):
    self.set_attr(self.AttrName.CONSTANT_VALUES, [int(constant)] * 4)


class TorchMatmul(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchMatmul, self).__init__(NNDCT_OP.MATMUL, *args, **kwargs)
    utils.op_register(NNDCT_OP.MATMUL, 'matmul')


class TorchClamp(Operation):

  def __init__(self, *args, **kwargs):
    super(TorchClamp, self).__init__(NNDCT_OP.CLAMP, *args, **kwargs)
    utils.op_register(NNDCT_OP.CLAMP, 'clamp')


class TorchSlice(base_op.StridedSlice):

  def __init__(self, input_ndim):
    super().__init__()
    self._input_ndim = input_ndim
    utils.op_register(NNDCT_OP.STRIDED_SLICE, NNDCT_OP.STRIDED_SLICE)
    
  @property
  def start(self):
    if self._input_ndim != 4:
      return self.get_attr(self.AttrName.BEGIN)
    else:
      begin = [0] * 4
      for dim, pos in enumerate(self.get_attr(self.AttrName.BEGIN)):
        new_dim = _transformed_axis(
            src="NHWC", dst="NCHW", ndim=self._input_ndim, dim=dim)
        begin[new_dim] = pos
      return begin

  @start.setter
  def start(self, start):
    if self._input_ndim != 4:
      begin_mask = 0
      for dim, pos in enumerate(start):
        if pos == 0:
          begin_mask |= 1 << dim
      self.set_attr(self.AttrName.BEGIN_MASK, begin_mask)
      self.set_attr(self.AttrName.BEGIN, start)

    else:
      begin = [0] * 4
      begin_mask = 0
      for dim, pos in enumerate(start):
        new_dim = _transformed_axis(
            src="NCHW", dst="NHWC", ndim=self._input_ndim, dim=dim)
        begin[new_dim] = pos

      for dim, pos in enumerate(begin):
        if pos == 0:
          begin_mask |= 1 << dim

      self.set_attr(self.AttrName.BEGIN_MASK, begin_mask)
      self.set_attr(self.AttrName.BEGIN, begin)

  @property
  def end(self):
    if self._input_ndim != 4:
      return self.get_attr(self.AttrName.END)
    else:
      end = [NNDCT_CONSTANT.INT_MAX] * 4
      for dim, pos in enumerate(self.get_attr(self.AttrName.END)):
        new_dim = _transformed_axis(
            src="NHWC", dst="NCHW", ndim=self._input_ndim, dim=dim)
        end[new_dim] = pos
      return end

  @end.setter
  def end(self, end):
    if self._input_ndim != 4:
      end_mask = 0
      for dim, pos in enumerate(end):
        if pos >= NNDCT_CONSTANT.INT_MAX:
          end_mask |= 1 << dim
      self.set_attr(self.AttrName.END_MASK, end_mask)
      self.set_attr(self.AttrName.END, end)
    else:
      end = [NNDCT_CONSTANT.INT_MAX] * 4
      end_mask = 0
      for dim, pos in enumerate(end):
        new_dim = _transformed_axis(
            src="NCHW", dst="NHWC", ndim=self._input_ndim, dim=dim)
        end[new_dim] = pos

      for dim, pos in enumerate(end):
        if pos >= NNDCT_CONSTANT.INT_MAX:
          end_mask |= 1 << dim

      self.set_attr(self.AttrName.END_MASK, end_mask)
      self.set_attr(self.AttrName.END, end)

  @property
  def step(self):
    if self._input_ndim != 4:
      return self.get_attr(self.AttrName.STRIDES)
    else:
      strides = [1] * 4
      for dim, step in enumerate(self.get_attr(self.AttrName.STRIDES)):
        new_dim = _transformed_axis(
            src="NHWC", dst="NCHW", ndim=self._input_ndim, dim=dim)
        strides[new_dim] = step
      return strides

  @step.setter
  def step(self, steps):
    if self._input_ndim != 4:
      self.set_attr(self.AttrName.STRIDES, steps)
    else:
      strides = [1] * 4
      for dim, step in enumerate(steps):
        new_dim = _transformed_axis(
            src="NCHW", dst="NHWC", ndim=self._input_ndim, dim=dim)
        strides[new_dim] = step

      self.set_attr(self.AttrName.STRIDES, strides)


class TorchArange(Operation):

  def __init__(self):
    super().__init__(NNDCT_OP.ARANGE)
    utils.op_register(NNDCT_OP.ARANGE, 'arange')


# class TorchSlicedInplaceCopy(Operation):

#   def __init__(self):
#     super().__init__(NNDCT_OP.SLICE_TENSOR_INPLACE_COPY)


class TorchBaseOperation(Operation):
  def __init__(self, nndct_op_type, torch_op_type=None):
    super().__init__(nndct_op_type)
    if torch_op_type is not None:
      utils.op_register(nndct_op_type, torch_op_type)
