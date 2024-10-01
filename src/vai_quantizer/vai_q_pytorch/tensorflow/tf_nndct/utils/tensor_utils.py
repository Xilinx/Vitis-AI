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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from nndct_shared.utils.tensor_util import convert_parameter_tensor_format
from nndct_shared.utils.tensor_util import DataFormatMap
from nndct_shared.pruning import pruning_lib
from nndct_shared.base import FrameworkType
from tf_nndct.graph.ops import Tensor
from tf_nndct.utils import keras_utils

#_tf_layouts = DataFormatMap._parameter_format_map['tensorflow']
#_nndct_layouts = DataFormatMap._parameter_format_map['nndct']
#
#_tf_blob_layout = DataFormatMap._blob_format_map['tensorflow']
#_nndct_blob_layout = DataFormatMap._blob_format_map['nndct']

def tf_blob_format(ndim):
  blob_format = {
      'channels_first': {
          2: 'NH',
          3: 'NCL',
          4: 'NCHW',
          5: 'NCDHW'
      },
      'channels_last': {
          2: 'NH',
          3: 'NLC',
          4: 'NHWC',
          5: 'NDHWC'
      }
  }
  return blob_format[keras_utils.data_format()][ndim]

def transposeconv_weight_dim_trans(weight: np.ndarray):
  # the default weight in convlayer is HWIO
  # the weight saved in transpoed conv is HWOI
  # in order to set_weight for tf.transposeconv layer
  # we need to convert the last two dim
  if len(weight.shape) < 2:
    return weight
  index_list = [x for x in range(len(weight.shape))]
  index_list[-1] -= 1
  index_list[-2] += 1
  return weight.transpose(index_list)

def param_from_tf_numpy(name, ndarray):
  tensor = Tensor.from_numpy(name, ndarray)
  return tf_param_to_nndct(tensor)

def param_to_tf_numpy(tensor):
  t = Tensor.from_numpy(tensor.name, np.copy(tensor.data))
  nndct_param_to_tf(t)
  return t.data

def tf_param_to_nndct(tensor):
  return convert_parameter_tensor_format(tensor, FrameworkType.TENSORFLOW,
                                         FrameworkType.NNDCT)

def nndct_param_to_tf(tensor):
  return convert_parameter_tensor_format(tensor, FrameworkType.NNDCT,
                                         FrameworkType.TENSORFLOW)

def tf_blob_to_nndct(tensor):
  return transpose(tensor, tf_blob_format(tensor.ndim),
                   DataFormatMap.blob_format(FrameworkType.NNDCT, tensor.ndim))

def _transpose(tensor, src_layout, dst_layout):
  if src_layout == dst_layout:
    return

  axis = [src_layout.index(d) for d in dst_layout]
  return tensor.transpose(axis)

def transpose(tensor, src_layout, dst_layout):
  if not isinstance(tensor, Tensor):
    raise TypeError("'tensor' must be Tensor, but given {}".format(
        type(tensor)))

  return _transpose(tensor, src_layout, dst_layout)

def layer_weights_from_node(node):
  '''
  convert the nndct tensor format -> tf format(channal last)
  more in nndct_shared/utils/tensor_util.py
  '''
  weights = []
  for tensor in node.op.params.values():
    # In tf weight in transpose conv2D/3D diff from conv2D/3D
    # transpose(2D/3D): kernel_shape = kernel_size + (self.filters, input_dim)
    # Conv2D kernel_shape = kernel_size + (input_dim, self.filters)
    if pruning_lib.is_transpose_conv(node.op):
      # get the weight for tf transpose conv layer we need transpoe the O <-> I
      weights.append(transposeconv_weight_dim_trans(param_to_tf_numpy(tensor)))
    else:
      # for normal conv/linear
      weights.append(param_to_tf_numpy(tensor))
  return weights
