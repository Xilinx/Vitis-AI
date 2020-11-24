

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

def mmul(m, n, shift):
    #m=m.astype(np.int32)
    #n=n.astype(np.int32)
    k = np.matmul(m, n)
    k = absolute_shift(k, shift)
    #k = shift_op(k, shift)
    #k1 =np.dot(m, n)
    return k

def emul(m, n, shift):
    m=m.astype(np.int32)
    n=n.astype(np.int32)
    k = m*n
    res = absolute_shift(k, shift)
    return res

def add(a=0, b=0):
    res= a+b
    res=absolute_shift(res, 0)
    return res

def sub(a, b):
    res= a-b
    res=absolute_shift(res, 0)
    return res

def absolute_shift(x, pos, to='right', bitwidth=16):
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


def mapping_sigm(x,data,shift=8,flavor='fpga'):
    def __ele_map(x_ele):
        if (x_ele >> shift) >= 8:
            return 2 ** 15 - 1
        elif (x_ele >> shift) < -8:
            return 0
        elif flavor=='math':
            if x_ele >=0:
                if shift >= 7:
                    pos = (x_ele >> (shift - 7)) % 1024
                else:
                    pos = (x_ele << (7 - shift)) % 1024
                return data[pos+1024]
            else:
                if shift >= 7:
                    pos = (abs(x_ele) >> (shift - 7)) % 1024
                else:
                    pos = (abs(x_ele) << (7 - shift)) % 1024
                if x_ele >> shift == -8 and pos == 0:
                    return 0
                else:
                    return data[1024-pos]
        elif flavor=='fpga':
            if shift >= 7:
                pos = (int(x_ele) >> (shift - 7)) % 2048
            else:
                pos = (int(x_ele) << (7 - shift)) % 2048
            return data[pos]
        else:
            raise TypeError("unexpected flavor: "+str(flavor))
    return np.array([[__ele_map(c) for c in row] for row in x], dtype=np.int16)

def mapping_tanh(x,data,shift=8,flavor='fpga'):
    def __ele_map(x_ele):
        if (x_ele >> shift) >= 4:
            return 2 ** 15 - 1
        elif (x_ele >> shift) < -4:
            return -2 ** 15
        elif flavor=='math':
            if x_ele >= 0:
                if shift >= 8:
                    pos = (x_ele >> (shift - 8)) % 1024
                else:
                    pos = (x_ele << (8 - shift)) % 1024
                return data[pos + 1024]
            else:
                if shift >= 8:
                    pos = (abs(x_ele) >> (shift - 8)) % 1024
                else:
                    pos = (abs(x_ele) << (8 - shift)) % 1024
                if x_ele >> shift == -4 and pos == 0:
                    return data[pos]
                else:
                    return data[1024-pos]
        elif flavor=='fpga':
            if shift >= 8:
                pos = (int(x_ele) >> (shift - 8)) % 2048
            else:
                pos = (int(x_ele) << (8 - shift)) % 2048
            return data[pos]
        else:
            raise TypeError("unexpected flavor: "+str(flavor))
    return np.array([[__ele_map(c) for c in row] for row in x], dtype=np.int16)

