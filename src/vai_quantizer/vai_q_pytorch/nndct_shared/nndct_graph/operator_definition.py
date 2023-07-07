

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

from enum import auto, unique, Enum
from typing import Any
from nndct_shared.base import NNDCT_OP
from nndct_shared.utils.common import AutoName
from nndct_shared.nndct_graph.base_operator import (NndctIrAttr,
                                                    OccurenceType, Operation)
from nndct_shared.nndct_graph.base_tensor import Tensor
import numpy as np

class Conv1d(Operation):

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
    super(Conv1d, self).__init__(*args, **kwargs)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.KERNEL: [None],
        self.AttrName.STRIDE: [None],
        self.AttrName.DILATION: [None],
        self.AttrName.PAD_MODE: [None],
        self.AttrName.PAD: [None, None],
        self.AttrName.GROUP: [None],
        self.AttrName.BIAS_TERM: [None],
        self.AttrName.IN_DIM: [None],
        self.AttrName.OUT_DIM: [None],
    }
    self._attrs[self.AttrName.KERNEL] = NndctIrAttr(
        name=self.AttrName.KERNEL,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.KERNEL],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""kernel size, [kernel_w, kernel_h]""")

    self._attrs[self.AttrName.STRIDE] = NndctIrAttr(
        name=self.AttrName.STRIDE,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.STRIDE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""stride [stride_w, stride_h]""")

    self._attrs[self.AttrName.DILATION] = NndctIrAttr(
        name=self.AttrName.DILATION,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.DILATION],
        occurence_type=OccurenceType.OPTIONAL,
        default_value=[1],
        annotation=r"""dilation, [dilation_w, dilation_h]""")

    self._attrs[self.AttrName.PAD_MODE] = NndctIrAttr(
        name=self.AttrName.PAD_MODE,
        value_type=str,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.PAD_MODE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""padding mode, 0-PADDING, 1-SAME, 2-VALID, 3-CEIL
    for the FUTURE. use attr pad. SAME, make output with same
    width and height as input. VALID, no padding""")

    self._attrs[self.AttrName.PAD] = NndctIrAttr(
        name=self.AttrName.PAD,
        value_type=int,
        size=2,
        value_mem=self._attr_value_mem[self.AttrName.PAD],
        occurence_type=OccurenceType.OPTIONAL,
        default_value=[0,0],
        annotation=r"""padding size, only effective when pad mode is PADDING, ["
                "left, right, top, bottom],""")

    self._attrs[self.AttrName.GROUP] = NndctIrAttr(
        name=self.AttrName.GROUP,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.GROUP],
        occurence_type=OccurenceType.OPTIONAL,
        default_value=1,
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
        annotation=r"""padding mode, 0-PADDING, 1-SAME, 2-VALID, 3-CEIL
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
        default_value=1,
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


class Conv3d(Operation):

  @unique
  class AttrName(AutoName):
    KERNEL = auto()
    STRIDE = auto()
    DILATION = auto()
    GROUP = auto()
    PAD_MODE = auto()
    PAD = auto()
    BIAS_TERM = auto()
    IN_DIM = auto()
    OUT_DIM = auto()
    OUTPUT_PAD = auto()

  @unique
  class ParamName(AutoName):
    WEIGHTS = auto()
    BIAS = auto()

  def __init__(self, *args, **kwargs) -> None:
    super(Conv3d, self).__init__(*args, **kwargs)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.KERNEL: [None, None, None],
        self.AttrName.STRIDE: [None, None, None],
        self.AttrName.DILATION: [None, None, None],
        self.AttrName.GROUP: [None],
        self.AttrName.PAD_MODE: [None],
        self.AttrName.PAD: [None, None, None, None, None, None],
        self.AttrName.OUTPUT_PAD: [None, None, None, None, None, None],
        self.AttrName.BIAS_TERM: [None],
        self.AttrName.IN_DIM: [None],
        self.AttrName.OUT_DIM: [None],
    }
    self._attrs[self.AttrName.KERNEL] = NndctIrAttr(
        name=self.AttrName.KERNEL,
        value_type=int,
        size=3,
        value_mem=self._attr_value_mem[self.AttrName.KERNEL],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""kernel size, [kernel_w, kernel_h, kernel_d]""")

    self._attrs[self.AttrName.STRIDE] = NndctIrAttr(
        name=self.AttrName.STRIDE,
        value_type=int,
        size=None,
        value_mem=self._attr_value_mem[self.AttrName.STRIDE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""stride [stride_w, stride_h, stride_d]""")

    self._attrs[self.AttrName.DILATION] = NndctIrAttr(
        name=self.AttrName.DILATION,
        value_type=int,
        size=None,
        value_mem=self._attr_value_mem[self.AttrName.DILATION],
        occurence_type=OccurenceType.OPTIONAL,
        default_value=[1, 1, 1],
        annotation=r"""dilation, [dilation_w, dilation_h, dilation_d]""")

    self._attrs[self.AttrName.GROUP] = NndctIrAttr(
        name=self.AttrName.GROUP,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.GROUP],
        occurence_type=OccurenceType.OPTIONAL,
        default_value=1,
        annotation=r"""group""")

    self._attrs[self.AttrName.PAD_MODE] = NndctIrAttr(
        name=self.AttrName.PAD_MODE,
        value_type=str,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.PAD_MODE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""We support 4 padding mode: `FLOOR, CEIL, SAME, VALID`. "
        "For example, when you parsing models from other frameworks, "
        "`caffe, pytorch->\"FLOOR\", tensorflow->\"SAME\" or \"VALID\"`""")

    self._attrs[self.AttrName.PAD] = NndctIrAttr(
        name=self.AttrName.PAD,
        value_type=int,
        size=6,
        value_mem=self._attr_value_mem[self.AttrName.PAD],
        occurence_type=OccurenceType.OPTIONAL,
        default_value=[0, 0, 0, 0, 0, 0],
        annotation=r"""padding size, only effective when pad mode is PADDING, ["
                "left, right, top, bottom, near, far],""")

    self._attrs[self.AttrName.OUTPUT_PAD] = NndctIrAttr(
        name=self.AttrName.OUTPUT_PAD,
        value_type=int,
        size=6,
        value_mem=self._attr_value_mem[self.AttrName.OUTPUT_PAD],
        occurence_type=OccurenceType.OPTIONAL,
        default_value=[0, 0, 0, 0, 0, 0],
        annotation=r"""additional size added to one side of each dimension in the output, ["
                "left, right, top, bottom, near, far],""")

    self._attrs[self.AttrName.BIAS_TERM] = NndctIrAttr(
        name=self.AttrName.BIAS_TERM,
        value_type=bool,
        size=None,
        value_mem=self._attr_value_mem[self.AttrName.BIAS_TERM],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""whether bias exist""")

    self._attrs[self.AttrName.IN_DIM] = NndctIrAttr(
        name=self.AttrName.IN_DIM,
        value_type=int,
        size=None,
        value_mem=self._attr_value_mem[self.AttrName.IN_DIM],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""in_channels""")

    self._attrs[self.AttrName.OUT_DIM] = NndctIrAttr(
        name=self.AttrName.OUT_DIM,
        value_type=int,
        size=None,
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
    AXIS = auto()

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
        self.AttrName.AXIS: [None]
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

    self._attrs[self.AttrName.AXIS] = NndctIrAttr(
        name=self.AttrName.AXIS,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.AXIS],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""the axis of the input to implement batchnorm""")


class InstanceNorm(Operation):

  @unique
  class AttrName(AutoName):
    EPS = auto()
    NUM_FEATURES = auto()
    AFFINE = auto()

  @unique
  class ParamName(AutoName):
    GAMMA = auto()
    BETA = auto()

  def __init__(self, *args, **kwargs) -> None:
    super(InstanceNorm, self).__init__(*args, **kwargs)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.EPS: [None],
        self.AttrName.NUM_FEATURES: [None],
        self.AttrName.AFFINE: [None],
    }
    self._attrs[self.AttrName.EPS] = NndctIrAttr(
        name=self.AttrName.EPS,
        value_type=float,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.EPS],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""eps""")

    self._attrs[self.AttrName.NUM_FEATURES] = NndctIrAttr(
        name=self.AttrName.NUM_FEATURES,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.NUM_FEATURES],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""num_features""")

    self._attrs[self.AttrName.AFFINE] = NndctIrAttr(
        name=self.AttrName.AFFINE,
        value_type=bool,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.AFFINE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""affine""")


class GroupNorm(Operation):

  @unique
  class AttrName(AutoName):
    NUM_GROUPS = auto()
    NUM_CHANNELS = auto()
    EPS = auto()
    AFFINE = auto()

  @unique
  class ParamName(AutoName):
    GAMMA = auto()
    BETA = auto()

  def __init__(self, *args, **kwargs) -> None:
    super(GroupNorm, self).__init__(*args, **kwargs)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.NUM_GROUPS: [None],
        self.AttrName.NUM_CHANNELS: [None],
        self.AttrName.EPS: [None],
        self.AttrName.AFFINE: [None]
    }
    self._attrs[self.AttrName.EPS] = NndctIrAttr(
        name=self.AttrName.EPS,
        value_type=float,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.EPS],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""eps""")

    self._attrs[self.AttrName.NUM_GROUPS] = NndctIrAttr(
        name=self.AttrName.NUM_GROUPS,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.NUM_GROUPS],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""num_groups""")

    self._attrs[self.AttrName.AFFINE] = NndctIrAttr(
        name=self.AttrName.AFFINE,
        value_type=bool,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.AFFINE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""affine""")

    self._attrs[self.AttrName.NUM_CHANNELS] = NndctIrAttr(
        name=self.AttrName.NUM_CHANNELS,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.NUM_CHANNELS],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""num_channels""")


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
        annotation=r"""padding mode, 0-PADDING, 1-SAME, 2-VALID, 3-CEIL
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
        default_value=False,
        annotation=r"""global""")

class MaxPool1d(Operation):

  @unique
  class AttrName(AutoName):
    KERNEL = auto()
    STRIDE = auto()
    DILATION = auto()
    PAD_MODE = auto()
    PAD = auto()
    GLOBAL = auto()

  def __init__(self, *args, **kwargs) -> None:
    super(MaxPool1d, self).__init__(*args, **kwargs)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.KERNEL: [None],
        self.AttrName.STRIDE: [None],
        self.AttrName.PAD_MODE: [None],
        self.AttrName.PAD: [None, None],
        self.AttrName.GLOBAL: [None],
    }
    self._attrs[self.AttrName.KERNEL] = NndctIrAttr(
        name=self.AttrName.KERNEL,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.KERNEL],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""kernel size, [kernel_w, kernel_h]""")

    self._attrs[self.AttrName.STRIDE] = NndctIrAttr(
        name=self.AttrName.STRIDE,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.STRIDE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""stride [stride_w, stride_h]""")

    self._attrs[self.AttrName.PAD_MODE] = NndctIrAttr(
        name=self.AttrName.PAD_MODE,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.PAD_MODE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""padding mode, 0-PADDING, 1-SAME, 2-VALID, 3-CEIL
    for the FUTURE. use attr pad. SAME, make output with same
    width and height as input. VALID, no padding""")

    self._attrs[self.AttrName.PAD] = NndctIrAttr(
        name=self.AttrName.PAD,
        value_type=int,
        size=2,
        value_mem=self._attr_value_mem[self.AttrName.PAD],
        occurence_type=OccurenceType.OPTIONAL,
        default_value=[0, 0],
        annotation=r"""padding size, only effective when pad mode is PADDING, ["
                "left, right, top, bottom],""")

    self._attrs[self.AttrName.GLOBAL] = NndctIrAttr(
        name=self.AttrName.GLOBAL,
        value_type=bool,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.GLOBAL],
        occurence_type=OccurenceType.OPTIONAL,
        default_value=False,
        annotation=r"""global""")


class AvgPool(Operation):

  @unique
  class AttrName(AutoName):
    KERNEL = auto()
    STRIDE = auto()
    PAD_MODE = auto()
    PAD = auto()
    GLOBAL = auto()
    COUNT_INCLUDE_PAD = auto()

  def __init__(self, *args, **kwargs) -> None:
    super(AvgPool, self).__init__(*args, **kwargs)

    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.KERNEL: [None, None],
        self.AttrName.STRIDE: [None, None],
        self.AttrName.PAD_MODE: [None],
        self.AttrName.PAD: [None, None, None, None],
        self.AttrName.GLOBAL: [None],
        self.AttrName.COUNT_INCLUDE_PAD: [None]
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
        annotation=r"""padding mode, 0-PADDING, 1-SAME, 2-VALID, 3-CEIL
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
        default_value=False,
        annotation=r"""global""")

    self._attrs[self.AttrName.COUNT_INCLUDE_PAD] = NndctIrAttr(
        name=self.AttrName.COUNT_INCLUDE_PAD,
        value_type=bool,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.COUNT_INCLUDE_PAD],
        occurence_type=OccurenceType.OPTIONAL,
        default_value=[True],
        annotation=r"""when True, will include the zero-padding in the averaging calculation""")


class Flatten(Operation):

  @unique
  class AttrName(AutoName):
    START_DIM = "start_axis"
    END_DIM = "end_axis"

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


# including mean, max, min etc.
class PermuteInvariantOp(Operation):

  @unique
  class AttrName(AutoName):
    DIMS = "axis"
    KEEP_DIMS = auto()

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.DIMS: [],
        self.AttrName.KEEP_DIMS: [None],
    }
    self._attrs[self.AttrName.DIMS] = NndctIrAttr(
        name=self.AttrName.DIMS,
        value_type=int,
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

  def __init__(self, op_type, *args, **kwargs) -> None:
    super(Permute, self).__init__(op_type)
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

  def __init__(self, *args, **kwargs) -> None:
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
        name=self.AttrName.INPUT_SIZE,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.INPUT_SIZE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""input size of GRU.""")

    self._attrs[self.AttrName.HIDDEN_SIZE] = NndctIrAttr(
        name=self.AttrName.HIDDEN_SIZE,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.HIDDEN_SIZE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""hidden size of GRU.""")

    self._attrs[self.AttrName.BIDIRECTIONAL] = NndctIrAttr(
        name=self.AttrName.BIDIRECTIONAL,
        value_type=bool,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.BIDIRECTIONAL],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r""" If True, means a bidirectional GRU.""")

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


class StridedSlice(Operation):

  @unique
  class AttrName(AutoName):
    DIMS = auto()
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
        self.AttrName.DIMS: [],
        self.AttrName.BEGIN: [],
        self.AttrName.END: [],
        self.AttrName.STRIDES: [],
        self.AttrName.BEGIN_MASK: [None],
        self.AttrName.END_MASK: [None],
        self.AttrName.ELLIPSIS_MASK: [None],
        self.AttrName.NEW_AXIS_MASK: [None],
        self.AttrName.SHRINK_AXIS_MASK: [None]
    }

    self._attrs[self.AttrName.DIMS] = NndctIrAttr(
        name=self.AttrName.DIMS,
        value_type=(int,Tensor),
        size=None,
        value_mem=self._attr_value_mem[self.AttrName.DIMS],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""start location of slicing (included)""",
        map_to_xir=False)


    self._attrs[self.AttrName.BEGIN] = NndctIrAttr(
        name=self.AttrName.BEGIN,
        value_type=(int,Tensor),
        size=None,
        value_mem=self._attr_value_mem[self.AttrName.BEGIN],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""start location of slicing (included)""")

    self._attrs[self.AttrName.END] = NndctIrAttr(
        name=self.AttrName.END,
        value_type=(int,Tensor),
        size=None,
        value_mem=self._attr_value_mem[self.AttrName.END],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""end location of slicing (excluded)""")

    self._attrs[self.AttrName.STRIDES] = NndctIrAttr(
        name=self.AttrName.STRIDES,
        value_type=(int,Tensor),
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


class BinaryOp(Operation):

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
        value_type=(int, float, bool, Tensor, np.ndarray),
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.INPUT],
        occurence_type=OccurenceType.REQUIRED,
        map_to_xir=False,
        annotation=r"""the first input tensor.""")

    self._attrs[self.AttrName.OTHER] = NndctIrAttr(
        name=self.AttrName.OTHER,
        value_type=(int, float, Tensor, np.ndarray),
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.OTHER],
        occurence_type=OccurenceType.REQUIRED,
        map_to_xir=False,
        annotation=r"""the second input tensor.""")


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
        value_type=(int, float, Tensor, np.ndarray),
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.INPUT],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""the first input tensor.""")

    self._attrs[self.AttrName.OTHER] = NndctIrAttr(
        name=self.AttrName.OTHER,
        value_type=(int, float, Tensor, np.ndarray),
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.OTHER],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""the second input tensor.""")


class Pad(Operation):

  @unique
  class AttrName(AutoName):
    PAD_WITH = "paddings"
    MODE = auto()
    CONSTANT_VALUES = auto()

  def __init__(self) -> None:
    super().__init__(NNDCT_OP.PAD)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.PAD_WITH: [None, None, None, None, None, None, None, None],
        self.AttrName.MODE: [None],
        self.AttrName.CONSTANT_VALUES: [None, None, None, None, None, None, None, None]
    }
    self._attrs[self.AttrName.PAD_WITH] = NndctIrAttr(
        name=self.AttrName.PAD_WITH,
        value_type=int,
        size=8,
        value_mem=self._attr_value_mem[self.AttrName.PAD_WITH],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""0 , 0 , left, right, top, bottom, 0, 0""")

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
        value_type=float,
        size=8,
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


class GELU(Operation):

  @unique
  class AttrName(AutoName):
    APPROXIMATE = auto()

  def __init__(self) -> None:
    super().__init__(NNDCT_OP.GELU)
    self._attr_value_mem = {
         self.AttrName.APPROXIMATE: [None],
     }
    self._attrs[self.AttrName.APPROXIMATE] = NndctIrAttr(
         name=self.AttrName.APPROXIMATE,
         value_type=str,
         size=1,
         value_mem=self._attr_value_mem[self.AttrName.APPROXIMATE],
         occurence_type=OccurenceType.REQUIRED,
         annotation=r"""the gelu approximation algorithm to use: 'none' | 'tanh'. Default: 'none'""")


class PReLU(Operation):

  @unique
  class AttrName(AutoName):
    NUM_PARAMETERS= auto()

  @unique
  class ParamName(AutoName):
    WEIGHT = auto()

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(NNDCT_OP.PRELU, *args, **kwargs)
    # allocate memory for attr value
    self._attr_value_mem = {
         self.AttrName.NUM_PARAMETERS: [None],
     }
    self._attrs[self.AttrName.NUM_PARAMETERS] = NndctIrAttr(
         name=self.AttrName.NUM_PARAMETERS,
         value_type=int,
         size=1,
         value_mem=self._attr_value_mem[self.AttrName.NUM_PARAMETERS],
         occurence_type=OccurenceType.REQUIRED,
         annotation=r"""number of a to learn""")


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
        value_type=str,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.MODE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""OPENCV-NEAREST -> 0, OPENCV-BILINEAR -> 1,
                Tensorflow-NEAREST -> 2, Tensorflow-BILINEAR -> 3,
                To be improved!""")

class Resize3d(Operation):

  @unique
  class AttrName(AutoName):
    SIZE = auto()
    SCALE = auto()
    ALIGN_CORNERS = auto()
    HALF_PIXEL_CENTERS = auto()
    MODE = auto()

  def __init__(self) -> None:
    super().__init__(NNDCT_OP.RESIZE_3D)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.SIZE: [None, None, None],
        self.AttrName.SCALE: [None, None, None],
        self.AttrName.ALIGN_CORNERS: [None],
        self.AttrName.HALF_PIXEL_CENTERS: [None],
        self.AttrName.MODE: [None],
    }

    self._attrs[self.AttrName.SIZE] = NndctIrAttr(
        name=self.AttrName.SIZE,
        value_type=(int, Tensor),
        size=3,
        value_mem=self._attr_value_mem[self.AttrName.SIZE],
        default_value=[0, 0, 0],
        occurence_type=OccurenceType.OPTIONAL,
        annotation=r"""output spatial size, [size_h, size_w, size_d]""")

    self._attrs[self.AttrName.SCALE] = NndctIrAttr(
        name=self.AttrName.SCALE,
        value_type=float,
        size=3,
        value_mem=self._attr_value_mem[self.AttrName.SCALE],
        default_value=[1.0, 1.0, 1.0],
        occurence_type=OccurenceType.OPTIONAL,
        annotation=r"""New size = Origin size * scale. {scale_h, scale_w, scale_d}.""")

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
        value_type=str,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.MODE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""Trilinear""")

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
        value_type=(int, float, list, Tensor, np.ndarray),
        size=None,
        value_mem=self._attr_value_mem[self.AttrName.DATA],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""Constant Parameter""")


class Squeeze(Operation):

  @unique
  class AttrName(AutoName):
    DIMS = "axis"

  def __init__(self) -> None:
    super().__init__(NNDCT_OP.SQUEEZE)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.DIMS: [],
    }

    self._attrs[self.AttrName.DIMS] = NndctIrAttr(
        name=self.AttrName.DIMS,
        value_type=int,
        size=None,
        value_mem=self._attr_value_mem[self.AttrName.DIMS],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""The dimensions to be squeezed. The dimension index "         //
                "starts at 0.""")


class EmbeddingBag(Operation):

  @unique
  class ParamName(AutoName):
    WEIGHT = auto()


class LayerNorm(Operation):

  @unique
  class AttrName(AutoName):
    EPS = auto()
    NORMALIZED_SHAPE = auto()
    ELEMENTWISE_AFFINE = auto()

  @unique
  class ParamName(AutoName):
    GAMMA = auto()
    BETA = auto()

  def __init__(self, *args, **kwargs) -> None:
    super(LayerNorm, self).__init__(*args, **kwargs)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.EPS: [None],
        self.AttrName.NORMALIZED_SHAPE: [None],
        self.AttrName.ELEMENTWISE_AFFINE: [None],
    }
    self._attrs[self.AttrName.EPS] = NndctIrAttr(
        name=self.AttrName.EPS,
        value_type=float,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.EPS],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""eps""")

    self._attrs[self.AttrName.NORMALIZED_SHAPE] = NndctIrAttr(
        name=self.AttrName.NORMALIZED_SHAPE,
        value_type=(int, Tensor, np.ndarray),
        size=None,
        value_mem=self._attr_value_mem[self.AttrName.NORMALIZED_SHAPE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""normalized_shape""")

    self._attrs[self.AttrName.ELEMENTWISE_AFFINE] = NndctIrAttr(
        name=self.AttrName.ELEMENTWISE_AFFINE,
        value_type=bool,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.ELEMENTWISE_AFFINE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""elementwise_affine""")


# e.g. ones, zeros
class ConstFromShape(Operation):

  @unique
  class AttrName(AutoName):
    SHAPE = auto()

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.SHAPE: [],

    }

    self._attrs[self.AttrName.SHAPE] = NndctIrAttr(
        name=self.AttrName.SHAPE,
        value_type=(int, Tensor),
        size=None,
        value_mem=self._attr_value_mem[self.AttrName.SHAPE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""the target shape""")


class UnaryOp(Operation):
  @unique
  class AttrName(AutoName):
    INPUT = auto()

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.INPUT: [None],
    }
    self._attrs[self.AttrName.INPUT] = NndctIrAttr(
        name=self.AttrName.INPUT,
        value_type=(int, str, float, bool, Tensor, np.ndarray),
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.INPUT],
        occurence_type=OccurenceType.REQUIRED,
        map_to_xir=False,
        annotation=r"""the first input tensor.""")


class Reorg(Operation):

  @unique
  class AttrName(AutoName):
    SCALE = auto()
    REVERSE = auto()

  def __init__(self, nndct_op_type) -> None:
    super().__init__(nndct_op_type)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.SCALE: [None],
        self.AttrName.REVERSE: [None],
    }

    self._attrs[self.AttrName.SCALE] = NndctIrAttr(
        name=self.AttrName.SCALE,
        value_type=(int, Tensor),
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.SCALE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""scale for reorg""")

    self._attrs[self.AttrName.REVERSE] = NndctIrAttr(
        name=self.AttrName.REVERSE,
        value_type=bool,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.REVERSE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""reverse""")


class Gstiling(Operation):

  @unique
  class AttrName(AutoName):
    STRIDE = auto()
    REVERSE = auto()

  def __init__(self, nndct_op_type) -> None:
    super().__init__(nndct_op_type)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.STRIDE: [None],
        self.AttrName.REVERSE: [None],
    }

    self._attrs[self.AttrName.STRIDE] = NndctIrAttr(
        name=self.AttrName.STRIDE,
        value_type=(int, Tensor),
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.STRIDE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""stride for feature maps""")

    self._attrs[self.AttrName.REVERSE] = NndctIrAttr(
        name=self.AttrName.REVERSE,
        value_type=bool,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.REVERSE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""reverse""")



class PixelShuffle(Operation):

  @unique
  class AttrName(AutoName):
    SCALE = auto()
    UPSCALE = auto()

  def __init__(self, nndct_op_type) -> None:
    super().__init__(nndct_op_type)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.SCALE: [None],
        self.AttrName.UPSCALE: [None],
    }

    self._attrs[self.AttrName.SCALE] = NndctIrAttr(
        name=self.AttrName.SCALE,
        value_type=(int, Tensor),
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.SCALE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""scale for feature maps""")

    self._attrs[self.AttrName.UPSCALE] = NndctIrAttr(
        name=self.AttrName.UPSCALE,
        value_type=bool,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.UPSCALE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""upscale or downscale PixelShuffle.""")

class Embedding(Operation):
  @unique
  class ParamName(AutoName):
    WEIGHT = auto()


class CustomOp(Operation):

  AttrName = Enum("AttrName", '')


  def __init__(self, nndct_op_type) -> None:
    super().__init__(nndct_op_type)
    self._attr_value_mem = {}
    self.is_custom_op = True


  def get_attr_name_from_str(self, attr_name):
    attr_names = [(name, val.value) for name, val in self.AttrName.__members__.items()]
    if(not attr_names) or (attr_names and all([attr_name != attr[1] for attr in attr_names])):
      attr_names += [(attr_name.upper(), attr_name.lower())]
      self.AttrName = Enum("AttrName", attr_names)
    return getattr(self.AttrName, attr_name.upper())

  def _register_attr_by_name(self, attr_name):
    if attr_name in self.AttrName.__members__:
      return

    attr_name = self.get_attr_name_from_str(attr_name)
    self._attr_value_mem[attr_name] = [None]
    self._attrs[attr_name] = NndctIrAttr(
        name=attr_name,
        value_type=Any,
        size=None,
        occurence_type=OccurenceType.REQUIRED,
        value_mem=self._attr_value_mem[attr_name])

  def set_attr_by_name(self, attr_name, value):
    if attr_name not in self.AttrName.__members__:
      self._register_attr_by_name(attr_name)
    attr_name = self.get_attr_name_from_str(attr_name)
    self.set_attr(attr_name, value)


class Correlation(Operation):
  @unique
  class AttrName(AutoName):
    PAD_SIZE = auto()

  # def __init__(self, nndct_op_type) -> None:
  #   super().__init__(nndct_op_type)
  def __init__(self, *args, **kwargs) -> None:
    super(Correlation, self).__init__(*args, **kwargs)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.PAD_SIZE: [None],
    }

    self._attrs[self.AttrName.PAD_SIZE] = NndctIrAttr(
        name=self.AttrName.PAD_SIZE,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.PAD_SIZE],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""pad size""")


class CostVolume(Operation):
  @unique
  class AttrName(AutoName):
    MAXDISP = auto()

  # def __init__(self, nndct_op_type) -> None:
  #   super().__init__(nndct_op_type)
  def __init__(self, *args, **kwargs) -> None:
    super(CostVolume, self).__init__(*args, **kwargs)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.MAXDISP: [None],
    }

    self._attrs[self.AttrName.MAXDISP] = NndctIrAttr(
        name=self.AttrName.MAXDISP,
        value_type=int,
        size=1,
        value_mem=self._attr_value_mem[self.AttrName.MAXDISP],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""max disp""")

class LogSoftmax(Operation):

  @unique
  class AttrName(AutoName):
    AXIS = auto()

  def __init__(self) -> None:
    super(LogSoftmax, self).__init__(NNDCT_OP.LOG_SOFTMAX)
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
        annotation=r"""the dimension log_softmax would be performed on. default
      is the last dimension.""")


class ArgMax_DIM(Operation):

  @unique
  class AttrName(AutoName):
    AXIS = auto()

  def __init__(self) -> None:
    super(ArgMax_DIM, self).__init__(NNDCT_OP.ARGMAX_DIM)
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
        annotation=r"""the dimension argmax would be performed on. default
      is the last dimension.""")


class Matmul(Operation):

  @unique
  class AttrName(AutoName):
    TRANSPOSE_A = auto()
    TRANSPOSE_B = auto()

  def __init__(self, *args, **kwargs) -> None:
    super(Matmul, self).__init__(NNDCT_OP.MATMUL)
    # allocate memory for attr value
    self._attr_value_mem = {
        self.AttrName.TRANSPOSE_A: [None],
        self.AttrName.TRANSPOSE_B: [None]
    }
    self._attrs[self.AttrName.TRANSPOSE_A] = NndctIrAttr(
        name=self.AttrName.TRANSPOSE_A,
        value_type=bool,
        size=1,
        default_value=False,
        value_mem=self._attr_value_mem[self.AttrName.TRANSPOSE_A],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""transpose_a of matmul.""")
    self._attrs[self.AttrName.TRANSPOSE_B] = NndctIrAttr(
        name=self.AttrName.TRANSPOSE_B,
        value_type=bool,
        size=1,
        default_value=False,
        value_mem=self._attr_value_mem[self.AttrName.TRANSPOSE_B],
        occurence_type=OccurenceType.REQUIRED,
        annotation=r"""transpose_b of matmul.""")
