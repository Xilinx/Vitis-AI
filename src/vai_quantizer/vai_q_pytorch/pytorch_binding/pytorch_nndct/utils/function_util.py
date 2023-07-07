

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

import torch
import numpy as np
from enum import Enum, unique

@unique
class Const(Enum):
  EPS = 0.0000152587890625

# 1/sqrt(x): input numpy
def invsqrt(number):
  x2 = number.astype(np.float32)
  x2 = x2*0.5
  y = number.astype(np.float32)
  threehalfs = 1.5
  i = y.view(np.int32)
  i = 0x5f3759df - (i >> 1)
  y = i.view(np.float32)
  y = y*(threehalfs - (x2*y*y))
  y = y*(threehalfs - (x2*y*y))
  y = y*(threehalfs - (x2*y*y))
  y = y*(threehalfs - (x2*y*y))
  return y

# simulate aie_reduce_add_v8
def aie_add_v8(v8): # input: tensor (L, 8)
  v4 = v8[:, 0:4] + v8[:, 4:]
  v2 = v4[:, 0:2] + v4[:, 2:]
  v1 = v2[:, 0] + v2[:, 1]
  return v1  # output: tensor (L,)

# simulate aie_reduce_add_v16
def aie_add_v16(v16): # input: tensor (L, 16)
  v8 = v16[:, 0:8] + v16[:, 8:]
  v4 = v8[:, 0:4] + v8[:, 4:]
  v2 = v4[:, 0:2] + v4[:, 2:]
  v1 = v2[:, 0] + v2[:, 1]
  return v1  # output: tensor (L,)

