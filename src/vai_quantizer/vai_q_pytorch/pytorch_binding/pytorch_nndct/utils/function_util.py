

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

# aie kernel: sqrt(x)=x*(1/sqrt(x)) with Newton iteration for 1/sqrt(x)
def sqrt(input): 
  x = input.to(torch.float32) # input tensor, float32
  x2 = x*0.5 # float32
  magic = 0x5f37
  y = np.float32(x.cpu().detach().numpy()) # float32
  i = y.view(np.int32)  # int32
  i = (magic - np.int32(i >> 17)) << 16 # int32
  y = i.view(np.float32)                # int32 to float32
  y = torch.from_numpy(y).to(x.device).to(x.dtype) # float32, initial value of Newton iteration 

  #  one step Newton iteration: y = y*(1.5 - (x2*y*y)) = 1.5*y - x2*y*y*y
  y3h = 1.5*y  # float32
  y3h = y3h.to(torch.bfloat16).to(torch.float32) # float32 to bfloat16 to float32

  out = y*x2  # float32
  out = out.to(torch.bfloat16).to(torch.float32) # float32 to bfloat32 to float32

  out = out*y
  out = out.to(torch.bfloat16).to(torch.float32) # float32 to bfloat32 to float32

  out = out*y
  out = out.to(torch.bfloat16).to(torch.float32) # float32 to bfloat32 to float32

  out = y3h - out
  out = out.to(torch.bfloat16).to(torch.float32) # float32 to bfloat32 to float32

  # sqrt(x) = x*(1/sqrt(x))
  out = x*out
  out = out.to(torch.bfloat16).to(torch.float32) # float32 to bfloat32 to float32
  
  return out

# aie kernel: 1/sqrt(x) with Newton iteration
def isqrt(input):
  from bfloat16 import bfloat16

  def downshift_onebit(i): # input: int16, numpy ndarray
    x = (i >> 1).reshape(-1)
    b = (i&1).reshape(-1)
    y = x
    for j in range(len(x)):
      if x[j]&1 == 1: # odd
        y[j] = y[j] + b[j]
    y = y.reshape(i.shape)
    return y

  input_np = input.detach().cpu().numpy()
  x = input_np.astype(bfloat16).astype(np.float32) # input tensor, bfloat16
  x2 = x*0.5
  x2 = x2.astype(bfloat16).astype(np.float32)

  y = input_np.astype(bfloat16) # bfloat16
  i = y.view(np.int16)  # int16
  i = 0x5f37 - downshift_onebit(i)
  y = i.view(bfloat16)
  y = y.astype(np.float32)

  # 4-steps-Newton iteration: y = y*(1.5 - (x2*y*y))
  for i in range(4):
    y2 = y*y
    y2 = y2.astype(bfloat16).astype(np.float32)

    mul2 = x2*y2
    mul2 = mul2.astype(bfloat16).astype(np.float32)

    sub = 1.5 - mul2
    sub = sub.astype(bfloat16).astype(np.float32)

    mul = y*sub
    y = mul.astype(bfloat16).astype(np.float32)

  return torch.from_numpy(y).to(input.device)
      
# 1/sqrt(x): input numpy
def invsqrt(number):
  x2 = number.astype(np.float32)
  x2 = x2 * 0.5
  y  = number.astype(np.float32)
  threehalfs = 1.5
  i = y.view(np.int32)
  i  = 0x5f3759df - (i >> 1)
  y = i.view(np.float32)
  y = y * (threehalfs - (x2 * y * y))
  y = y * (threehalfs - (x2 * y * y))
  y = y * (threehalfs - (x2 * y * y))
  y = y * (threehalfs - (x2 * y * y))
  
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

