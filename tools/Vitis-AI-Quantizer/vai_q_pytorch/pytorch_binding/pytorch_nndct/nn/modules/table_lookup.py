

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

def mysigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def mapping_sigm(x, data, shift):
    scale = 1.0 / 2 ** 15
    inv_step = 2 ** shift
    def __ele_map(x_ele):
        scale = 2 ** -15
        if x_ele >= 8:
            return 1.0 - scale
        elif x_ele < -8:
            return 0.0
        else:
            x_ele = int(x_ele * inv_step)
            if x_ele >=0:
                if shift >= 7:
                    pos = (x_ele >> (shift - 7)) % 1024
                else:
                    pos = (x_ele << (7 - shift)) % 1024
                return data[pos+1024] * scale
            else:
                if shift >= 7:
                    pos = (abs(x_ele) >> (shift - 7)) % 1024
                else:
                    pos = (abs(x_ele) << (7 - shift)) % 1024
                if x_ele >> shift == -8 and pos == 0:
                    return 0.0
                else:
                    return data[1024-pos] * scale
    return np.array([[__ele_map(c) for c in row] for row in x], dtype=np.float32)

def mapping_tanh(x, data, shift):
    scale = 1.0 / 2 ** 15
    inv_step = 2 ** shift
    def __ele_map(x_ele):
        if x_ele >= 4:
            return 1.0 - scale
        elif x_ele < -4:
            return -1.0
        else:
            x_ele = int(x_ele * inv_step)
            if x_ele >= 0:
                if shift >= 8:
                    pos = (x_ele >> (shift - 8)) % 1024
                else:
                    pos = (x_ele << (8 - shift)) % 1024
                return data[pos + 1024] * scale
            else:
                if shift >= 8:
                    pos = (abs(x_ele) >> (shift - 8)) % 1024
                else:
                    pos = (abs(x_ele) << (8 - shift)) % 1024
                if x_ele >> shift == -4 and pos == 0:
                    return data[pos] * scale
                else:
                    return data[1024-pos] * scale
    return np.array([[__ele_map(c) for c in row] for row in x], dtype=np.float32)

def absolute_shift(x, pos, to='left', bitwidth=16):
    res = 0
    if to == 'left':
        if pos >= 0:
            res = np.left_shift(x, pos)
        else:
            res = np.right_shift(x, -pos)
    elif to == 'right':
        if pos >= 0:
            res = np.right_shift(x, pos)
        else:
            res = np.left_shift(x, -pos)
    else:
        raise TypeError("shift to {} is not expected".format(to))
    res = np.where(res > (2**(bitwidth - 1) - 1), 2**(bitwidth - 1) - 1, res)
    res = np.where(res < -2**(bitwidth - 1), -2**(bitwidth - 1), res)

    return res

def absolute_shift_round(x, pos, to='left', bitwidth=16):
    res = 0
    if to == 'left':
        if pos >= 0:
            #res = np.left_shift(x, pos)
            res = x * (2**pos)
        else:
            #res = np.right_shift(x, -pos)
            res = x * (2**(-pos))
    elif to == 'right':
        if pos >= 0:
            #res = np.right_shift(x, pos)
            res = np.round(x / (2**pos)).astype(np.int32)
        else:
            res = np.round(x / (2**(-pos))).astype(np.int32)
    else:
        raise TypeError("shift to {} is not expected".format(to))
    res = np.where(res > (2**(bitwidth - 1) - 1), 2**(bitwidth - 1) - 1, res)
    res = np.where(res < -2**(bitwidth - 1), -2**(bitwidth - 1), res)

    return res
