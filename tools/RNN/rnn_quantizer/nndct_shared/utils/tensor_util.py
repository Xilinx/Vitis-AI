

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

"""Utility functions for manipulating NndctTensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from nndct_shared.base.key_names import NNDCT_OP
from nndct_shared.base.key_names import FrameworkType
from nndct_shared.nndct_graph import base_tensor


class DataFormat(object):
  channel_first = "channel first"
  channel_last = "channel_last"


class DataFormatMap(object):
  """A dict mapping of framework and op type to its data format.
  """

  _blob_format_map = {
      FrameworkType.NNDCT: {
          2: "NH",
          3: "NLC",
          4: "NHWC",
          5: "NHWDC"
      },
      FrameworkType.TORCH: {
          2: "NH",
          3: "NCL",
          4: "NCHW",
          5: "NCDHW"
      },
      # TF format generated in runtime.
  }

  _parameter_format_map = {
      FrameworkType.NNDCT: {
          2: "OI",
          3: "OLI",
          4: "OHWI",
          5: "OHWDI"
      },
      FrameworkType.TORCH: {
          2: "OI",
          3: "OIL",
          4: "OIHW",
          5: "OIDHW"
      },
      FrameworkType.TENSORFLOW: {
          2: "IO",
          3: "LIO",
          4: "HWIO",
          5: "DHWIO",
      }
  }


  @classmethod
  def blob_format(cls, framework_type, ndim):
    if framework_type not in cls._blob_format_map:
      raise KeyError(
          "Framework type '{}' not supported now.".format(framework_type))
    return cls._blob_format_map[framework_type][ndim]

  @classmethod
  def param_format(cls, framework_type, ndim):
    if framework_type not in cls._parameter_format_map:
      raise KeyError(
          "Framework type '{}' not supported now.".format(framework_type))
    return cls._parameter_format_map[framework_type][ndim]


def layout_transformer(src_layout, dst_layout):
  assert len(src_layout) == len(dst_layout)
  axes = []
  for axis in dst_layout:
    axes.append(src_layout.index(axis))
  return tuple(axes)

def convert_blob_tensor_format(tensor: base_tensor.Tensor, src_framework: str,
                               dst_framework: str) -> base_tensor.Tensor:
  if not isinstance(tensor, base_tensor.Tensor):
    raise TypeError("'tensor' must be Tensor, but given {}".format(
        type(tensor)))
  if not tensor.is_complete_tensor():
    return tensor

  if src_framework == dst_framework:
    return tensor

  if tensor.ndim not in DataFormatMap._blob_format_map[src_framework].keys():
    return tensor

  src_layout = DataFormatMap.blob_format(src_framework, tensor.ndim)
  dst_layout = DataFormatMap.blob_format(dst_framework, tensor.ndim)

  if src_layout == dst_layout:
    return tensor

  tensor.transpose(layout_transformer(src_layout, dst_layout))

  return tensor

def convert_parameter_tensor_format(tensor: base_tensor.Tensor,
                                    src_framework: str,
                                    dst_framework: str) -> base_tensor.Tensor:
  if not isinstance(tensor, base_tensor.Tensor):
    raise TypeError("'tensor' must be Tensor, but given {}".format(
        type(tensor)))
  if not tensor.is_complete_tensor():
    return tensor

  if src_framework == dst_framework:
    return tensor

  if tensor.ndim not in DataFormatMap._parameter_format_map[src_framework].keys():
    return tensor

  src_format = DataFormatMap.param_format(src_framework, tensor.ndim)
  dst_format = DataFormatMap.param_format(dst_framework, tensor.ndim)

  if src_format == dst_format:
    return tensor

  tensor.transpose(layout_transformer(src_format, dst_format))

  return tensor


def transformed_axis(src: str, dst: str, ndim: int, dim: int) -> int:
  """NC* -> N*C/ N*C ->NC*"""
  if ndim is None or ndim not in [4, 5] or src == dst:
    return dim
  # NCHW -> NHWC / NHWC - > NCHW
  if ndim == 4:
    if src == DataFormat.channel_first and dst == DataFormat.channel_last:
      return dim + [0, 2, -1, -1][dim]
    elif src == DataFormat.channel_last and dst == DataFormat.channel_first:
      return dim + [0, 1, 1, -2][dim]

  # NCDHW -> NHWDC / NHWDC -> NCDHW
  elif ndim == 5:
    if src == DataFormat.channel_first and dst == DataFormat.channel_last:
      return dim + [0, 3, 1, -2, -2][dim]
    elif src == DataFormat.channel_last and dst == DataFormat.channel_first:
      return dim + [0, 2, 2, -1, -3][dim]


def permute_data(data, order):
  if order is None or (not isinstance(data, np.ndarray)):
    return data

  if len(order) != data.ndim:
    raise RuntimeError("The data dimensions should consistent with length of order")

  return np.transpose(data, order)


def permute_axes(axes, order):
  if order is None:
    return axes
  new_axes = [None] * len(axes)
  for i, j in enumerate(order):
    new_axes[i] = axes[j]

  return new_axes


def combine_orders(order1, order2):
  new_order = len(order1) * [None]
  for i in range(len(order1)):
    t_i = order1.index(i)
    new_order[i] = order2.index(t_i)

  return new_order
