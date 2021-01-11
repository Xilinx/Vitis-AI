

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

class DataFormatMap(object):
  """A dict mapping of framework and op type to its data format.
  """
  _nndct_format_map = {
      NNDCT_OP.CONV2D: 'OHWI',
      NNDCT_OP.DEPTHWISE_CONV2D: 'OHWI',
      NNDCT_OP.CONVTRANSPOSE2D: 'OHWI',
      NNDCT_OP.DENSE: 'OI',
      NNDCT_OP.BASIC_LSTM: 'OI'
  }

  _torch_format_map = {
      NNDCT_OP.CONV2D: 'OIHW',
      NNDCT_OP.DEPTHWISE_CONV2D: 'OIHW',
      NNDCT_OP.CONVTRANSPOSE2D: 'OIHW',
      NNDCT_OP.DENSE: 'OI',
      NNDCT_OP.BASIC_LSTM: 'OI'
  }

  _tf_format_map = {
      NNDCT_OP.CONV2D: 'HWIO',
      NNDCT_OP.DEPTHWISE_CONV2D: 'HWIO',
      NNDCT_OP.DENSE: 'IO',
      NNDCT_OP.BASIC_LSTM: 'IO'
  }

  _param_format_map = {
      FrameworkType.NNDCT: _nndct_format_map,
      FrameworkType.TORCH: _torch_format_map,
      FrameworkType.TENSORFLOW: _tf_format_map,
      FrameworkType.TF_KERAS: _tf_format_map,
  }

  _blob_format_map = {
      FrameworkType.NNDCT: "NHWC",
      FrameworkType.TORCH: "NCHW",
  }

  _parameter_format_map = {
      FrameworkType.NNDCT: {
          4: "OHWI",
          2: "OI"
      },
      FrameworkType.TORCH: {
          4: "OIHW",
          2: "OI"
      },
  }

  @classmethod
  def framework_formats(cls, framework_type):
    if framework_type not in cls._param_format_map:
      raise KeyError(
          "Framework type '{}' not supported now.".format(framework_type))
    return cls._param_format_map[framework_type]

  # @staticmethod
  # def nndct_output_formats(op_type: str)-> str:
  #   return 'NHWC'
  @classmethod
  def blob_format(cls, framework_type):
    if framework_type not in cls._blob_format_map:
      raise KeyError(
          "Framework type '{}' not supported now.".format(framework_type))
    return cls._blob_format_map[framework_type]

  @classmethod
  def param_format(cls, framework_type, ndim):
    if framework_type not in cls._parameter_format_map:
      raise KeyError(
          "Framework type '{}' not supported now.".format(framework_type))
    return cls._parameter_format_map[framework_type][ndim]

def convert_blob_tensor_format(tensor: base_tensor.Tensor, src_framework: str,
                               dst_framework: str) -> base_tensor.Tensor:
  if not isinstance(tensor, base_tensor.Tensor):
    raise TypeError("'tensor' must be Tensor, but given {}".format(
        type(tensor)))
  if not tensor.is_complete_tensor():
    return tensor
  if tensor.ndim != 4:
    return tensor
  if src_framework == dst_framework:
    return tensor

  src_layout = DataFormatMap.blob_format(src_framework)
  dst_layout = DataFormatMap.blob_format(dst_framework)

  if src_layout == dst_layout:
    return tensor
  if src_layout == "NCHW" and dst_layout == "NHWC":
    tensor.transpose((0, 2, 3, 1))
  elif src_layout == "NHWC" and dst_layout == "NCHW":
    tensor.transpose((0, 3, 1, 2))
  else:
    raise ValueError(
        "Can not transpose data format of '{}' from '{}' to '{}'".format(
            op_type, src_layout, dst_layout))
  return tensor

def convert_parameter_tensor_format(tensor: base_tensor.Tensor,
                                    src_framework: str,
                                    dst_framework: str) -> base_tensor.Tensor:
  if not isinstance(tensor, base_tensor.Tensor):
    raise TypeError("'tensor' must be Tensor, but given {}".format(
        type(tensor)))
  if not tensor.is_complete_tensor():
    return tensor
  if tensor.ndim != 4 and tensor.ndim != 2:
    return tensor
  if src_framework == dst_framework:
    return tensor

  src_format = DataFormatMap.param_format(src_framework, tensor.ndim)
  dst_format = DataFormatMap.param_format(dst_framework, tensor.ndim)

  if src_format == dst_format:
    return tensor
  elif src_format == 'OIHW' and dst_format == 'HWIO':
    tensor.transpose((2, 3, 1, 0))
  elif src_format == 'HWIO' and dst_format == 'OIHW':
    tensor.transpose((3, 2, 0, 1))
  elif src_format == 'OIHW' and dst_format == 'OWHI' or \
      (src_format == 'OWHI' and dst_format == 'OIHW'):
    tensor.transpose((0, 3, 2, 1))
  elif src_format == 'HWIO' and dst_format == 'OWHI':
    tensor.transpose((3, 1, 0, 2))
  elif src_format == 'OIHW' and dst_format == 'OHWI':
    tensor.transpose((0, 2, 3, 1))
  elif src_format == 'OHWI' and dst_format == 'OIHW':
    tensor.transpose((0, 3, 1, 2))
  elif (src_format == 'OI' and dst_format == 'IO') or \
      (src_format == 'IO' and dst_format == 'OI'):
    tensor.transpose((1, 0))
  else:
    raise ValueError("Can not transpose data format from '{}' to '{}'".format(
        src_format, dst_format))
  return tensor
