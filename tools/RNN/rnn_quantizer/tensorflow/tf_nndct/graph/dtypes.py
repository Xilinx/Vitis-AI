

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

import numpy as np

from enum import Enum

from tensorflow.core.framework import types_pb2

class DTypes(str, Enum):
  FLOAT = 'float',
  FLOAT16 = 'float16',
  DOUBLE = 'double',
  INT32 = 'int32',
  INT16 = 'int16',
  INT8 = 'int8',
  UINT8 = 'uint8',
  QUINT8 = 'quint8',
  UINT16 = 'uint16',
  INT64 = 'int64',
  UINT64 = 'uint64',
  STRING = 'string',
  COMPLEX64 = 'complex64',
  COMPLEX128 = 'complex128',
  BOOL = 'bool'

def from_numpy(dtype):
  return _NP_TO_NNDCT[dtype]

def to_numpy(dtype):
  return _NNDCT_TO_NP[dtype]

# mapping dtypes from nndct to numpy
_NNDCT_TO_NP = {
    DTypes.FLOAT: np.float32,
    DTypes.FLOAT16: np.float16,
    DTypes.DOUBLE: np.float64,
    DTypes.INT32: np.int32,
    DTypes.INT16: np.int16,
    DTypes.INT8: np.int8,
    DTypes.UINT8: np.uint8,
    DTypes.UINT16: np.uint16,
    DTypes.INT64: np.int64,
    DTypes.UINT64: np.uint64,
    DTypes.BOOL: np.bool,
}

def from_tf(dtype):
  return _TF_TO_NNDCT[dtype]

def to_tf(dtype):
  return _NNDCT_TO_TF[dtype]

# mapping dtypes from tensorflow to nndct
_TF_TO_NNDCT = {
    types_pb2.DT_FLOAT: DTypes.FLOAT,
    types_pb2.DT_HALF: DTypes.FLOAT16,
    types_pb2.DT_DOUBLE: DTypes.DOUBLE,
    types_pb2.DT_INT32: DTypes.INT32,
    types_pb2.DT_INT16: DTypes.INT16,
    types_pb2.DT_INT8: DTypes.INT8,
    types_pb2.DT_UINT8: DTypes.UINT8,
    types_pb2.DT_UINT16: DTypes.UINT16,
    types_pb2.DT_INT64: DTypes.INT64,
    types_pb2.DT_STRING: DTypes.STRING,
    types_pb2.DT_COMPLEX64: DTypes.COMPLEX64,
    types_pb2.DT_COMPLEX128: DTypes.COMPLEX128,
    types_pb2.DT_BOOL: DTypes.BOOL,
    types_pb2.DT_QUINT8: DTypes.QUINT8
}

_NNDCT_TO_TF = {nndct: tf for tf, nndct in _TF_TO_NNDCT.items()}
