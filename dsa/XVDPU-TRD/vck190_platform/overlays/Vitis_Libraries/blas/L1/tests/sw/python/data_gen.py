# Copyright 2019 Xilinx, Inc.
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

import numpy as np
import os
import math
import argparse
from operation import DataGenerator


class gemm_dataGenerator:

    def __init__(self, dirname, dataType):
        self.dataGen = DataGenerator()
        if dataType == 'float':
            self.dataGen.setDataType(np.float32)
            self.dtype = np.float32
            self.dataGen.setRange(-100, 100)
        elif dataType == 'short' or dataType == 'int16_t':
            self.dataGen.setDataType(np.int16)
            self.dtype = np.int16
            self.dataGen.setRange(-100, 100)
        elif dataType == 'int' or dataType == 'int32_t':
            self.dataGen.setDataType(np.int32)
            self.dtype = np.int32
            self.dataGen.setRange(-100, 100)
        elif dataType == 'int8_t':
            self.dataGen.setDataType(np.int8)
            self.dtype = np.int8
            self.dataGen.setRange(-100, 100)
        elif dataType == 'uint8_t':
            self.dataGen.setDataType(np.uint8)
            self.dtype = np.uint8
            self.dataGen.setRange(0, 100)
        elif dataType == 'uint16_t':
            self.dataGen.setDataType(np.uint16)
            self.dtype = np.uint16
            self.dataGen.setRange(0, 100)
        elif dataType == 'uint32_t':
            self.dataGen.setDataType(np.uint32)
            self.dtype = np.uint32
            self.dataGen.setRange(0, 100)
        else:
            print('dataType not supported')
        self.dirname = dirname

    def compute(self, args):
        os.makedirs(self.dirname, exist_ok=True)
        size_m = args.m
        size_n = args.n
        size_k = args.k
        matA = self.dataGen.matrix((size_m, size_k))
        matTA = np.transpose(matA)
        matTA.tofile(os.path.join(self.dirname, 'matA.bin'))
        matB = self.dataGen.matrix((size_k, size_n))
        matB.tofile(os.path.join(self.dirname, 'matB.bin'))
        matC = self.dataGen.matrix((size_m, size_n))
        matC.tofile(os.path.join(self.dirname, 'matC.bin'))
        matC = (
            args.alpha *
            np.matmul(
                matA,
                matB,
                dtype=self.dtype) +
            args.beta *
            matC).astype(
            self.dtype)
        matC.tofile(os.path.join(self.dirname, 'golden.bin'))


def main(args):
    if args.func == 'gemm':
        gemm = gemm_dataGenerator(args.dirname, args.dataType)
        gemm.compute(args)
    else:
        print('func not supported')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate random vectors and run test.')
    parser.add_argument('--m', type=int)
    parser.add_argument('--k', type=int)
    parser.add_argument('--n', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--beta', type=float)
    parser.add_argument(
        '--dirname',
        type=str,
        default="./data",
        help='data bin file dirname')
    parser.add_argument('--dataType', type=str, default='float')
    parser.add_argument('--func', type=str, default='gemm')

    args = parser.parse_args()
    main(args)
