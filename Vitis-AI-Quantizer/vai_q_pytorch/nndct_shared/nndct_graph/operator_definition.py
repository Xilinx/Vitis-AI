from enum import auto, unique

from nndct_shared.base import NNDCT_OP
from nndct_shared.nndct_graph.base_operator import (AutoName, NndctIrAttr,
                                                    OccurenceType, Operation)
from nndct_shared.nndct_graph.base_tensor import Tensor


class Conv2d(Operation):

  @unique
  class AttrName(AutoName):
    KERNEL = auto()
    STRIDE = auto()
    DILATION = auto()
    PAD_MODE = auto()
    PAD = auto()
    GROUP = auto()
    BIAS_TERM = auto()
    IN_DIM = auto()
    OUT_DIM = auto()

  @unique
  class ParamName(AutoName):
    WEIGHTS = auto()
    BIAS = auto()

  def __init__(self, *args, **kwargs) -> None:
    super(Conv2d, self).__init__(*args, **kwargs)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.KERNEL: [None, None],
        self.AttrName.STRIDE: [None, None],
        self.AttrName.DILATION: [None, None],
        self.AttrName.PAD_MODE: [None],
        self.AttrName.PAD: [None, None, None, None],
        self.AttrName.GROUP: [None],
        self.AttrName.BIAS_TERM: [None],
        self.AttrName.IN_DIM: [None],
        self.AttrName.OUT_DIM: [None],
    }
    self._attrs[self.AttrName.KERNEL] = NndctIrAttr(
        name=self.AttrName.KERNEL,
        value_type=int,
        size=2,
        value_mem=self._attr_value_mem[self.AttrName.KERNEL],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""kernel size, [kernel_w, kernel_h]""")

    self._attrs[self.AttrName.STRIDE] = NndctIrAttr(
        name=self.AttrName.STRIDE,
        value_type=int,
        size=2,
        value_mem=self._attr_value_mem[self.AttrName.STRIDE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""stride [stride_w, stride_h]""")

    self._attrs[self.AttrName.DILATION] = NndctIrAttr(
        name=self.AttrName.DILATION,
        value_type=int,
        size=2,
        value_mem=self._attr_value_mem[self.AttrName.DILATION],
        occurence_type=OccurenceType.OPTIONAL,
        default_value=[1, 1],
        annotation=r"""dilation, [dilation_w, dilation_h]""")

    self._attrs[self.AttrName.PAD_MODE] = NndctIrAttr(
        name=self.AttrName.PAD_MODE,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.PAD_MODE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""padding mode, 0-PADDING, 1-SAME, 2-VALID, 3-MODE LEFT
    for the FUTURE. use attr pad. SAME, make output with same
    width and height as input. VALID, no padding""")

    self._attrs[self.AttrName.PAD] = NndctIrAttr(
        name=self.AttrName.PAD,
        value_type=int,
        size=4,
        value_mem=self._attr_value_mem[self.AttrName.PAD],
        occurence_type=OccurenceType.OPTIONAL,
        default_value=[0, 0, 0, 0],
        annotation=r"""padding size, only effective when pad mode is PADDING, ["
                "left, right, top, bottom],""")

    self._attrs[self.AttrName.GROUP] = NndctIrAttr(
        name=self.AttrName.GROUP,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.GROUP],
        occurence_type=OccurenceType.OPTIONAL,
        default_value=[1],
        annotation=r"""group""")

    self._attrs[self.AttrName.BIAS_TERM] = NndctIrAttr(
        name=self.AttrName.BIAS_TERM,
        value_type=bool,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.BIAS_TERM],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""whether bias exist""")

    self._attrs[self.AttrName.IN_DIM] = NndctIrAttr(
        name=self.AttrName.IN_DIM,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.IN_DIM],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""in_channels""")

    self._attrs[self.AttrName.OUT_DIM] = NndctIrAttr(
        name=self.AttrName.OUT_DIM,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.OUT_DIM],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""out_channels""")


class BatchNorm(Operation):

  @unique
  class AttrName(AutoName):
    EPSILON = auto()
    SCALE = auto()
    CENTER = auto()
    OUT_DIM = auto()

  @unique
  class ParamName(AutoName):
    GAMMA = auto()
    BETA = auto()
    MOVING_MEAN = auto()
    MOVING_VAR = auto()

  def __init__(self, *args, **kwargs) -> None:
    super(BatchNorm, self).__init__(*args, **kwargs)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.EPSILON: [None],
        self.AttrName.SCALE: [None],
        self.AttrName.CENTER: [None],
        self.AttrName.OUT_DIM: [None],
    }
    self._attrs[self.AttrName.EPSILON] = NndctIrAttr(
        name=self.AttrName.EPSILON,
        value_type=float,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.EPSILON],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""epsilon""")

    self._attrs[self.AttrName.SCALE] = NndctIrAttr(
        name=self.AttrName.SCALE,
        value_type=bool,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.SCALE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""scale""")

    self._attrs[self.AttrName.CENTER] = NndctIrAttr(
        name=self.AttrName.CENTER,
        value_type=bool,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.CENTER],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""center""")

    self._attrs[self.AttrName.OUT_DIM] = NndctIrAttr(
        name=self.AttrName.OUT_DIM,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.OUT_DIM],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""num features""")


class Dense(Operation):

  @unique
  class AttrName(AutoName):
    BIAS_TERM = auto()
    IN_DIM = auto()
    OUT_DIM = auto()

  @unique
  class ParamName(AutoName):
    WEIGHTS = auto()
    BIAS = auto()

  def __init__(self, *args, **kwargs) -> None:
    super(Dense, self).__init__(*args, **kwargs)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.BIAS_TERM: [None],
        self.AttrName.IN_DIM: [None],
        self.AttrName.OUT_DIM: [None],
    }

    self._attrs[self.AttrName.BIAS_TERM] = NndctIrAttr(
        name=self.AttrName.BIAS_TERM,
        value_type=bool,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.BIAS_TERM],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""whether bias exist""")

    self._attrs[self.AttrName.IN_DIM] = NndctIrAttr(
        name=self.AttrName.IN_DIM,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.IN_DIM],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""in_channels""")

    self._attrs[self.AttrName.OUT_DIM] = NndctIrAttr(
        name=self.AttrName.OUT_DIM,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.OUT_DIM],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""out_channels""")


class Concat(Operation):

  @unique
  class AttrName(AutoName):
    AXIS = auto()

  def __init__(self, *args, **kwargs) -> None:
    super(Concat, self).__init__(*args, **kwargs)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.AXIS: [None],
    }

    self._attrs[self.AttrName.AXIS] = NndctIrAttr(
        name=self.AttrName.AXIS,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.AXIS],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""specified axis""")


class Shape(Operation):

  @unique
  class AttrName(AutoName):
    AXIS = auto()

  def __init__(self, *args, **kwargs) -> None:
    super(Shape, self).__init__(*args, **kwargs)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.AXIS: [None],
    }

    self._attrs[self.AttrName.AXIS] = NndctIrAttr(
        name=self.AttrName.AXIS,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.AXIS],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""specified axis""")


class Reshape(Operation):

  @unique
  class AttrName(AutoName):
    SHAPE = auto()

  def __init__(self, *args, **kwargs) -> None:
    super(Reshape, self).__init__(*args, **kwargs)

    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.SHAPE: [],  # possible any length
    }

    self._attrs[self.AttrName.SHAPE] = NndctIrAttr(
        name=self.AttrName.SHAPE,
        value_type=(int, Tensor),
        size=None,
        value_mem=self._attr_value_mem[self.AttrName.SHAPE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""the target shape""")


class MaxPool(Operation):

  @unique
  class AttrName(AutoName):
    KERNEL = auto()
    STRIDE = auto()
    DILATION = auto()
    PAD_MODE = auto()
    PAD = auto()
    GLOBAL = auto()

  def __init__(self, *args, **kwargs) -> None:
    super(MaxPool, self).__init__(*args, **kwargs)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.KERNEL: [None, None],
        self.AttrName.STRIDE: [None, None],
        self.AttrName.PAD_MODE: [None],
        self.AttrName.PAD: [None, None, None, None],
        self.AttrName.GLOBAL: [None],
    }
    self._attrs[self.AttrName.KERNEL] = NndctIrAttr(
        name=self.AttrName.KERNEL,
        value_type=int,
        size=2,
        value_mem=self._attr_value_mem[self.AttrName.KERNEL],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""kernel size, [kernel_w, kernel_h]""")

    self._attrs[self.AttrName.STRIDE] = NndctIrAttr(
        name=self.AttrName.STRIDE,
        value_type=int,
        size=2,
        value_mem=self._attr_value_mem[self.AttrName.STRIDE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""stride [stride_w, stride_h]""")

    self._attrs[self.AttrName.PAD_MODE] = NndctIrAttr(
        name=self.AttrName.PAD_MODE,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.PAD_MODE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""padding mode, 0-PADDING, 1-SAME, 2-VALID, 3-MODE LEFT
    for the FUTURE. use attr pad. SAME, make output with same
    width and height as input. VALID, no padding""")

    self._attrs[self.AttrName.PAD] = NndctIrAttr(
        name=self.AttrName.PAD,
        value_type=int,
        size=4,
        value_mem=self._attr_value_mem[self.AttrName.PAD],
        occurence_type=OccurenceType.OPTIONAL,
        default_value=[0, 0, 0, 0],
        annotation=r"""padding size, only effective when pad mode is PADDING, ["
                "left, right, top, bottom],""")

    self._attrs[self.AttrName.GLOBAL] = NndctIrAttr(
        name=self.AttrName.GLOBAL,
        value_type=bool,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.GLOBAL],
        occurence_type=OccurenceType.OPTIONAL,
        default_value=[False],
        annotation=r"""global""")


class AvgPool(Operation):

  @unique
  class AttrName(AutoName):
    KERNEL = auto()
    STRIDE = auto()
    PAD_MODE = auto()
    PAD = auto()
    GLOBAL = auto()

  def __init__(self, *args, **kwargs) -> None:
    super(AvgPool, self).__init__(*args, **kwargs)

    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.KERNEL: [None, None],
        self.AttrName.STRIDE: [None, None],
        self.AttrName.PAD_MODE: [None],
        self.AttrName.PAD: [None, None, None, None],
        self.AttrName.GLOBAL: [None],
    }
    self._attrs[self.AttrName.KERNEL] = NndctIrAttr(
        name=self.AttrName.KERNEL,
        value_type=int,
        size=2,
        value_mem=self._attr_value_mem[self.AttrName.KERNEL],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""kernel size, [kernel_w, kernel_h]""")

    self._attrs[self.AttrName.STRIDE] = NndctIrAttr(
        name=self.AttrName.STRIDE,
        value_type=int,
        size=2,
        value_mem=self._attr_value_mem[self.AttrName.STRIDE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""stride [stride_w, stride_h]""")

    self._attrs[self.AttrName.PAD_MODE] = NndctIrAttr(
        name=self.AttrName.PAD_MODE,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.PAD_MODE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""padding mode, 0-PADDING, 1-SAME, 2-VALID, 3-MODE LEFT
    for the FUTURE. use attr pad. SAME, make output with same
    width and height as input. VALID, no padding""")

    self._attrs[self.AttrName.PAD] = NndctIrAttr(
        name=self.AttrName.PAD,
        value_type=int,
        size=4,
        value_mem=self._attr_value_mem[self.AttrName.PAD],
        occurence_type=OccurenceType.OPTIONAL,
        default_value=[0, 0, 0, 0],
        annotation=r"""padding size, only effective when pad mode is PADDING, ["
                "left, right, top, bottom],""")

    self._attrs[self.AttrName.GLOBAL] = NndctIrAttr(
        name=self.AttrName.GLOBAL,
        value_type=bool,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.GLOBAL],
        occurence_type=OccurenceType.OPTIONAL,
        default_value=[False],
        annotation=r"""global""")


class Flatten(Operation):

  @unique
  class AttrName(AutoName):
    START_DIM = auto()
    END_DIM = auto()

  def __init__(self, *args, **kwargs) -> None:
    super(Flatten, self).__init__(*args, **kwargs)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.START_DIM: [None],
        self.AttrName.END_DIM: [None],
    }
    self._attrs[self.AttrName.START_DIM] = NndctIrAttr(
        name=self.AttrName.START_DIM,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.START_DIM],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""the first dim to flatten""")

    self._attrs[self.AttrName.END_DIM] = NndctIrAttr(
        name=self.AttrName.END_DIM,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.END_DIM],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""the  last dim to flatten""")


class Mean(Operation):

  @unique
  class AttrName(AutoName):
    DIMS = auto()
    KEEP_DIMS = auto()

  def __init__(self, *args, **kwargs) -> None:
    super(Mean, self).__init__(*args, **kwargs)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.DIMS: [],
        self.AttrName.KEEP_DIMS: [None],
    }
    self._attrs[self.AttrName.DIMS] = NndctIrAttr(
        name=self.AttrName.DIMS,
        value_type=list,
        size=None,
        value_mem=self._attr_value_mem[self.AttrName.DIMS],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""The dimensions to reduce. List of integers""")

    self._attrs[self.AttrName.KEEP_DIMS] = NndctIrAttr(
        name=self.AttrName.KEEP_DIMS,
        value_type=bool,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.KEEP_DIMS],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""specify whether the reduced dimension is kept or not.""")


class Permute(Operation):

  @unique
  class AttrName(AutoName):
    ORDER = auto()

  def __init__(self, *args, **kwargs) -> None:
    super(Permute, self).__init__(*args, **kwargs)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.ORDER: [],
    }
    self._attrs[self.AttrName.ORDER] = NndctIrAttr(
        name=self.AttrName.ORDER,
        value_type=(int, Tensor),
        size=None,
        value_mem=self._attr_value_mem[self.AttrName.ORDER],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""The dimensions to reduce. List of integers""")


class Softmax(Operation):

  @unique
  class AttrName(AutoName):
    AXIS = auto()

  def __init__(self) -> None:
    super(Softmax, self).__init__(NNDCT_OP.SOFTMAX)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.AXIS: [None],
    }
    self._attrs[self.AttrName.AXIS] = NndctIrAttr(
        name=self.AttrName.AXIS,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.AXIS],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""the dimension softmax would be performed on. default
      is the last dimension.""")


class Lstm(Operation):

  @unique
  class AttrName(AutoName):
    INPUT_SIZE = auto()
    HIDDEN_SIZE = auto()
    BIDIRECTIONAL = auto()
    NUM_LAYERS = auto()
    BATCH_FIRST = auto()

  @unique
  class ParamName(AutoName):
    WEIGHT_IH = auto()
    WEIGHT_HH = auto()
    WEIGHT_IH_REVERSE = auto()
    WEIGHT_HH_REVERSE = auto()
    BIAS = auto()
    BIAS_REVERSE = auto()

  def __init__(self, *args, **kwargs) -> None:
    super(Lstm, self).__init__(*args, **kwargs)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.INPUT_SIZE: [None],
        self.AttrName.HIDDEN_SIZE: [None],
        self.AttrName.BIDIRECTIONAL: [None],
        self.AttrName.NUM_LAYERS: [None],
        self.AttrName.BATCH_FIRST: [None],
    }

    self._attrs[self.AttrName.INPUT_SIZE] = NndctIrAttr(
        name=self.AttrName.INPUT_SIZE,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.INPUT_SIZE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""input size of LSTM.""")

    self._attrs[self.AttrName.HIDDEN_SIZE] = NndctIrAttr(
        name=self.AttrName.HIDDEN_SIZE,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.HIDDEN_SIZE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""hidden size of LSTM.""")

    self._attrs[self.AttrName.BIDIRECTIONAL] = NndctIrAttr(
        name=self.AttrName.BIDIRECTIONAL,
        value_type=bool,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.BIDIRECTIONAL],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r""" If True, means a bidirectional LSTM.""")

    self._attrs[self.AttrName.NUM_LAYERS] = NndctIrAttr(
        name=self.AttrName.NUM_LAYERS,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.NUM_LAYERS],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""Number of recurrent layers""")

    self._attrs[self.AttrName.BATCH_FIRST] = NndctIrAttr(
        name=self.AttrName.BATCH_FIRST,
        value_type=bool,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.BATCH_FIRST],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r""" If True, then the input and output tensors are provided as (batch, seq, feature)"""
    )
  
class Gru(Operation):
  @unique
  class AttrName(AutoName):
    INPUT_SIZE = auto()
    HIDDEN_SIZE = auto()
    BIDIRECTIONAL = auto()
    NUM_LAYERS = auto()
    BATCH_FIRST = auto()
    

  @unique
  class ParamName(AutoName):
    WEIGHT_IH = auto()
    WEIGHT_HH = auto()
    WEIGHT_IH_REVERSE = auto()
    WEIGHT_HH_REVERSE = auto()
    BIAS_IH = auto()
    BIAS_HH = auto()
    BIAS_IH_REVERSE = auto()
    BIAS_HH_REVERSE = auto()

  def __init__(self, *args, **kwargs) ->None:
    super(Gru, self).__init__(*args, **kwargs)
    # allocate memory for attr value
    self._attr_value_mem = {
      self.AttrName.INPUT_SIZE: [None],
      self.AttrName.HIDDEN_SIZE: [None],
      self.AttrName.BIDIRECTIONAL: [None],
      self.AttrName.NUM_LAYERS: [None],
      self.AttrName.BATCH_FIRST: [None],
     
    }
    self._attrs[self.AttrName.INPUT_SIZE] = NndctIrAttr(
      name = self.AttrName.INPUT_SIZE,
      value_type = int,
      size = 1,
      value_mem = self._attr_value_mem[self.AttrName.INPUT_SIZE],
      occurence_type = OccurenceType.REQUIRED,
      annotation=r"""input size of GRU."""
    )
    
    self._attrs[self.AttrName.HIDDEN_SIZE] = NndctIrAttr(
      name = self.AttrName.HIDDEN_SIZE,
      value_type = int,
      size = 1,
      value_mem = self._attr_value_mem[self.AttrName.HIDDEN_SIZE],
      occurence_type = OccurenceType.REQUIRED,
      annotation=r"""hidden size of GRU."""
    )
    
    self._attrs[self.AttrName.BIDIRECTIONAL] = NndctIrAttr(
      name = self.AttrName.BIDIRECTIONAL,
      value_type = bool,
      size = 1,
      value_mem = self._attr_value_mem[self.AttrName.BIDIRECTIONAL],
      occurence_type = OccurenceType.REQUIRED,
      annotation=r""" If True, means a bidirectional GRU."""
    )

    self._attrs[self.AttrName.NUM_LAYERS] = NndctIrAttr(
      name = self.AttrName.NUM_LAYERS,
      value_type = int,
      size = 1,
      value_mem = self._attr_value_mem[self.AttrName.NUM_LAYERS],
      occurence_type = OccurenceType.REQUIRED,
      annotation=r"""Number of recurrent layers"""
    )

    self._attrs[self.AttrName.BATCH_FIRST] = NndctIrAttr(
      name = self.AttrName.BATCH_FIRST,
      value_type = bool,
      size = 1,
      value_mem = self._attr_value_mem[self.AttrName.BATCH_FIRST],
      occurence_type = OccurenceType.REQUIRED,
      annotation=r""" If True, then the input and output tensors are provided as (batch, seq, feature)"""
    )
    
class StridedSlice(Operation):

  @unique
  class AttrName(AutoName):
    BEGIN = auto()
    END = auto()
    STRIDES = auto()
    BEGIN_MASK = auto()
    END_MASK = auto()
    ELLIPSIS_MASK = auto()
    NEW_AXIS_MASK = auto()
    SHRINK_AXIS_MASK = auto()

  def __init__(self) -> None:
    super(StridedSlice, self).__init__(NNDCT_OP.STRIDED_SLICE)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.BEGIN: [],
        self.AttrName.END: [],
        self.AttrName.STRIDES: [],
        self.AttrName.BEGIN_MASK: [None],
        self.AttrName.END_MASK: [None],
        self.AttrName.ELLIPSIS_MASK: [None],
        self.AttrName.NEW_AXIS_MASK: [None],
        self.AttrName.SHRINK_AXIS_MASK: [None]
    }

    self._attrs[self.AttrName.BEGIN] = NndctIrAttr(
        name=self.AttrName.BEGIN,
        value_type=int,
        size=None,
        value_mem=self._attr_value_mem[self.AttrName.BEGIN],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""start location of slicing (included)""")

    self._attrs[self.AttrName.END] = NndctIrAttr(
        name=self.AttrName.END,
        value_type=int,
        size=None,
        value_mem=self._attr_value_mem[self.AttrName.END],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""end location of slicing (excluded)""")

    self._attrs[self.AttrName.STRIDES] = NndctIrAttr(
        name=self.AttrName.STRIDES,
        value_type=int,
        size=None,
        value_mem=self._attr_value_mem[self.AttrName.STRIDES],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""strides of slicing""")

    self._attrs[self.AttrName.BEGIN_MASK] = NndctIrAttr(
        name=self.AttrName.BEGIN_MASK,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.BEGIN_MASK],
        default_value=0,
        occurence_type=OccurenceType.OPTIONAL,
        annotation=r"""If the ith bit of begin_mask is set, begin[i] is ignored
                  and the fullest possible range in that dimension is used
                  instead.""")

    self._attrs[self.AttrName.END_MASK] = NndctIrAttr(
        name=self.AttrName.END_MASK,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.END_MASK],
        default_value=0,
        occurence_type=OccurenceType.OPTIONAL,
        annotation=r"""If the ith bit of end_mask is set, end[i] is ignored and
                  the fullest possible range in that dimension is used
                  instead, except with the end range.""")

    self._attrs[self.AttrName.ELLIPSIS_MASK] = NndctIrAttr(
        name=self.AttrName.ELLIPSIS_MASK,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.ELLIPSIS_MASK],
        default_value=0,
        occurence_type=OccurenceType.OPTIONAL,
        annotation=r"""If the ith bit of ellipsis_mask is set, as many
                unspecified dimensions as needed will be inserted between
                other dimensions. Only one non-zero bit is allowed in
                ellipsis_mask.""")

    self._attrs[self.AttrName.NEW_AXIS_MASK] = NndctIrAttr(
        name=self.AttrName.NEW_AXIS_MASK,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.NEW_AXIS_MASK],
        default_value=0,
        occurence_type=OccurenceType.OPTIONAL,
        annotation=r"""If the ith bit of new_axis_mask is set, then begin, end,
                    and stride are ignored and a new length 1 dimension is
                    added at this point in the output tensor.""")

    self._attrs[self.AttrName.SHRINK_AXIS_MASK] = NndctIrAttr(
        name=self.AttrName.SHRINK_AXIS_MASK,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.SHRINK_AXIS_MASK],
        default_value=0,
        occurence_type=OccurenceType.OPTIONAL,
        annotation=r"""If the ith bit of shrink_axis_mask is set, it implies that
                taking on the value at index begin[i]. end[i] and
                strides[i] are ignored in this case.""")


class Sub(Operation):

  @unique
  class AttrName(AutoName):
    INPUT = auto()
    OTHER = auto()

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.INPUT: [None],
        self.AttrName.OTHER: [None],
    }
    self._attrs[self.AttrName.INPUT] = NndctIrAttr(
        name=self.AttrName.INPUT,
        value_type=(int, float, Tensor),
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.INPUT],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""the first input tensor.""")

    self._attrs[self.AttrName.OTHER] = NndctIrAttr(
        name=self.AttrName.OTHER,
        value_type=(int, float, Tensor),
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.OTHER],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""the second input tensor.""")


class Pad(Operation):

  @unique
  class AttrName(AutoName):
    PAD_WITH = auto()
    MODE = auto()
    CONSTANT_VALUES = auto()

  def __init__(self) -> None:
    super().__init__(NNDCT_OP.PAD)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.PAD_WITH: [None, None, None, None],
        self.AttrName.MODE: [None],
        self.AttrName.CONSTANT_VALUES: [None, None, None, None]
    }
    self._attrs[self.AttrName.PAD_WITH] = NndctIrAttr(
        name=self.AttrName.PAD_WITH,
        value_type=int,
        size=4,
        value_mem=self._attr_value_mem[self.AttrName.PAD_WITH],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""left, right, top, bottom""")

    self._attrs[self.AttrName.MODE] = NndctIrAttr(
        name=self.AttrName.MODE,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.MODE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""The padding mode. 0:'CONSTANT', 1:'REFLECT', 2:'SYMMETRIC']."""
    )

    self._attrs[self.AttrName.CONSTANT_VALUES] = NndctIrAttr(
        name=self.AttrName.CONSTANT_VALUES,
        value_type=int,
        size=4,
        value_mem=self._attr_value_mem[self.AttrName.CONSTANT_VALUES],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""the value set into the padded locations""")


class LeakyReLU(Operation):

  @unique
  class AttrName(AutoName):
    ALPHA = auto()

  def __init__(self) -> None:
    super().__init__(NNDCT_OP.LEAKY_RELU)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.ALPHA: [None],
    }
    self._attrs[self.AttrName.ALPHA] = NndctIrAttr(
        name=self.AttrName.ALPHA,
        value_type=float,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.ALPHA],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""negative slope""")


class Resize(Operation):

  @unique
  class AttrName(AutoName):
    SIZE = auto()
    SCALE = auto()
    ALIGN_CORNERS = auto()
    HALF_PIXEL_CENTERS = auto()
    MODE = auto()

  def __init__(self) -> None:
    super().__init__(NNDCT_OP.RESIZE)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.SIZE: [None, None],
        self.AttrName.SCALE: [None, None],
        self.AttrName.ALIGN_CORNERS: [None],
        self.AttrName.HALF_PIXEL_CENTERS: [None],
        self.AttrName.MODE: [None],
    }

    self._attrs[self.AttrName.SIZE] = NndctIrAttr(
        name=self.AttrName.SIZE,
        value_type=(int, Tensor),
        size=2,
        value_mem=self._attr_value_mem[self.AttrName.SIZE],
        default_value=[0, 0],
        occurence_type=OccurenceType.OPTIONAL,
        annotation=r"""output spatial size, [size_w, size_h]""")

    self._attrs[self.AttrName.SCALE] = NndctIrAttr(
        name=self.AttrName.SCALE,
        value_type=float,
        size=2,
        value_mem=self._attr_value_mem[self.AttrName.SCALE],
        default_value=[1.0, 1.0],
        occurence_type=OccurenceType.OPTIONAL,
        annotation=r"""New size = Origin size * scale. {scale_w, scale_h}.""")

    self._attrs[self.AttrName.ALIGN_CORNERS] = NndctIrAttr(
        name=self.AttrName.ALIGN_CORNERS,
        value_type=bool,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.ALIGN_CORNERS],
        default_value=False,
        occurence_type=OccurenceType.OPTIONAL,
        annotation=r"""It must be set When mode is 3.If true, the centers of 
                the 4 corner pixels of the input and output tensors are    
                aligned, preserving the values at the corner pixels.      
                Defaults to false.""")

    self._attrs[self.AttrName.HALF_PIXEL_CENTERS] = NndctIrAttr(
        name=self.AttrName.HALF_PIXEL_CENTERS,
        value_type=bool,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.HALF_PIXEL_CENTERS],
        default_value=False,
        occurence_type=OccurenceType.OPTIONAL,
        annotation=r"""half_pixel_centers is false by default in,
                tf.resize_bilinear() and tf.resize_nearest_neighbor().
                is true by default in tf.upsampling2d(), but the version
                of tf should be > r1.13""")

    self._attrs[self.AttrName.MODE] = NndctIrAttr(
        name=self.AttrName.MODE,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.MODE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""OPENCV-NEAREST -> 0, OPENCV-BILINEAR -> 1,
                Tensorflow-NEAREST -> 2, Tensorflow-BILINEAR -> 3,
                To be improved!""")


class Constant(Operation):

  @unique
  class AttrName(AutoName):
    DATA = auto()

  def __init__(self, nndct_op_type) -> None:
    super().__init__(nndct_op_type)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.DATA: [],
    }

    self._attrs[self.AttrName.DATA] = NndctIrAttr(
        name=self.AttrName.DATA,
        value_type=(int, float, list),
        size=None,
        value_mem=self._attr_value_mem[self.AttrName.DATA],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""Constant Parameter""")
    
    
class Squeeze(Operation):

  @unique
  class AttrName(AutoName):
    DIMS = auto()

  def __init__(self) -> None:
    super().__init__(NNDCT_OP.SQUEEZE)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.DIMS: [],
    }

    self._attrs[self.AttrName.DIMS] = NndctIrAttr(
        name=self.AttrName.DIMS,
        value_type=(int),
        size=None,
        value_mem=self._attr_value_mem[self.AttrName.DIMS],
        default_value=[0],
        occurence_type=OccurenceType.OPTIONAL,
        annotation=r"""The dimensions to be squeezed. The dimension index "         //
                "starts at 0.""")
