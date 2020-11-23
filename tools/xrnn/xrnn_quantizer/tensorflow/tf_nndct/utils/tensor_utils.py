

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tf_nndct.graph.ops import Tensor

_tf_layouts = {2: 'IO', 4: 'HWIO'}
_nndct_layouts = {2: 'OI', 4: 'OHWI'}

_tf_blob_layout = 'NHWC'
_nndct_blob_layout = 'NHWC'

def from_tf_numpy(name, ndarray):
  tensor = Tensor.from_numpy(name, ndarray)
  return tf_to_nndct(tensor)

def to_tf_numpy(tensor):
  t = Tensor.from_numpy(tensor.name, np.copy(tensor.data))
  nndct_to_tf(t)
  return t.data

def tf_to_nndct(tensor):
  return transpose(tensor, _tf_layouts, _nndct_layouts)

def nndct_to_tf(tensor):
  return transpose(tensor, _nndct_layouts, _tf_layouts)

# TODO(yuwang): Merge to tf_to_nndct ?
def tf_blob_to_nndct(tensor):
  return _transpose(tensor, _tf_blob_layout, _nndct_blob_layout)

def _transpose(tensor, src_layout, dst_layout):
  if src_layout == dst_layout:
    return

  axis = [src_layout.index(d) for d in dst_layout]
  return tensor.transpose(axis)

def transpose(tensor, src_layouts, dst_layouts):
  if not isinstance(tensor, Tensor):
    raise TypeError("'tensor' must be Tensor, but given {}".format(
        type(tensor)))

  if tensor.ndim != 4 and tensor.ndim != 2:
    return tensor

  src_layout = src_layouts[tensor.ndim]
  dst_layout = dst_layouts[tensor.ndim]
  return _transpose(tensor, src_layout, dst_layout)
