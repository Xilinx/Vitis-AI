

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
import math

#for calibration process
def max(data, name='', quantizer=None):
  return data.max()

def min(data, name='', quantizer=None):
  return data.min()

def quant_diff_s(data, bitwidth, range, round_method=2, name='',
                 quantizer=None):
  raise NotImplementedError("please implement the diffs operation")

#for quantization process
def __amplify_data(data, max, amp, method=2):
  # method = -1 means to quantize a not quantized float tensor and converted it to integer tensor
  # method > 0 means to convert a quantized float tensor to integer tensor
  data = data * amp
  if method == -1:
    data = np.clip(data, -max, max - 1)
    data = np.array([math.floor(v + 0.5) if v >= 0.0 else 
                     math.ceil(v) if (v - math.floor(v) == 0.5) else
                     math.ceil(v - 0.5) 
                     for v in data])

  return data

def normal_quant_neuron(data,
                        maxamps=[[32768], [2048]],
                        strides=[-1],
                        round_method=2,
                        keep_scale=True,
                        name='',
                        quantizer=None,
                        on_gpu=True,
                        as_int=False):
  #integer need not keep scale as precondition
  if as_int:
    keep_scale = False
  if len(strides) == 1:
    data = __amplify_data(
        data, maxamps[0][0], maxamps[1][0], method=round_method)
    if keep_scale:
      data = data / maxamps[1][0]
  else:
    org_shape = data.shape
    flatten_data = data.flatten()
    pos = 0
    for idx, s in enumerate(strides):
      flatten_data[pos:pos + s] = __amplify_data(
          flatten_data[pos:pos + s],
          maxamps[0][idx],
          maxamps[1][idx],
          method=round_method)
      if keep_scale:
        flatten_data[pos:pos + s] = flatten_data[pos:pos + s] / maxamps[1][idx]
      pos += s
    data = flatten_data.reshape(org_shape)
  #return integer or origional dtype
  if as_int:
    assert all(m == maxamps[0][0]
               for m in maxamps[0]), "all max limitation should be the same"
    if maxamps[0][0] == 2**7 or maxamps[0][0] == 2**3:
      return data.astype(np.int8)
    elif maxamps[0][0] == 2**15:
      return data.astype(np.int16)
    else:
      raise TypeError("unexpected max found " + str(maxamps[0][0]))
  else:
    return data

def nonlin(data, alpha, signed):
  if signed:
    return np.clip(data, -alpha, alpha)
  else:
    return np.clip(data, 0, alpha)

def pact_quant_neuron(data,
                      bitw,
                      bita,
                      alpha_init_value=None,
                      signed=False,
                      trainable=True,
                      warmup=False,
                      name='',
                      tensor_type='act',
                      quantizer=None):
  raise NotImplementedError("please implement the pact_quant_neuron operation")

def graffitist_quant_neuron(data, bn, fp, method=2, name=''):
  raise NotImplementedError(
      "please implement the lowbit_quant_neuron operation")
