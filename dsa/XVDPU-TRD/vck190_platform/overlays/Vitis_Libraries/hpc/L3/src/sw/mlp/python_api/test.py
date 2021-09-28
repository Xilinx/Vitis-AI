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

import numpy as np
import sys
import math
import xfhpc_L3 as xfhpc
from operation import DataGenerator


class Test:

    def cmp(self, A, B):
        if np.array_equal(A, B):
            print("Success!\n")
        else:
            print("not equal!")
            np.savetxt("A.np", A, fmt="%d")
            np.savetxt("B.np", B, fmt="%d")
            sys.exit(1)

    def cmpWithinTolerance(self, A, B):
        if np.allclose(A, B, 1e-3, 1e-5):
            print("Success!\n")
        else:
            print(A.shape, B.shape)
            np.savetxt("C.np", A, fmt="%f")
            np.savetxt("C_cpu.np", B, fmt="%f")
            diff = np.isclose(A.flatten(), B.flatten(), 1e-3, 1e-5)
            countDiff = diff.shape[0] - np.count_nonzero(diff)
            print("not equal, number of mismatches = ", countDiff)
            mismatch = ((diff == 0).nonzero())
            print("mismatches are in ", mismatch[0])
            for i in mismatch[0]:
                print(A.flatten()[i], " is different from ", B.flatten()[i])
            sys.exit(1)

    def get_padded_size(self, size, min_size):
        size_padded = int(math.ceil(np.float32(size) / min_size) * min_size)
        return size_padded

    def test_basic_fcn(
            self,
            m,
            k,
            n,
            xclbin_opts,
            post_val=[
                1,
                0],
            pRelu_val=[
                1,
                0],
        idxKernel=0,
        idxDevice=0,
        minRange=-
        16384,
            maxRange=16384):
        if xclbin_opts['BLAS_dataType'] == 'float':
            dtype = np.float32
        else:
            raise TypeError(
                "type",
                xclbin_opts["BLAS_dataType"],
                "not supported")
        if xclbin_opts['BLAS_XdataType'] == 'float':
            Xdtype = np.float32
        else:
            raise TypeError(
                "type",
                xclbin_opts["BLAS_dataType"],
                "not supported")
        ddrWidth = int(xclbin_opts["BLAS_ddrWidth"])
        padded_m = self.get_padded_size(
            m, int(xclbin_opts["BLAS_gemmMBlocks"]) * ddrWidth)
        padded_k = self.get_padded_size(
            k, int(xclbin_opts["BLAS_gemmKBlocks"]) * ddrWidth)
        padded_n = self.get_padded_size(
            n, int(xclbin_opts["BLAS_gemmNBlocks"]) * ddrWidth)

        self.dataGen = DataGenerator()
        self.dataGen.setRange(minRange, maxRange)
        self.dataGen.setDataType(dtype)
        A = self.dataGen.matrix((padded_m, padded_k))
        B = self.dataGen.matrix((padded_k, padded_n))
        C = self.dataGen.matrix((padded_m, padded_n))
        self.dataGen.setDataType(Xdtype)
        X = self.dataGen.matrix((padded_m, padded_n))

        golden_C = (np.matmul(A, B, dtype=dtype) + X).astype(dtype)

        if pRelu_val == [1, 0]:
            golden_C = 1 / (1 + np.exp(-golden_C))
        xfhpc.sendMat(A, idxKernel, idxDevice)
        xfhpc.sendMat(B, idxKernel, idxDevice)
        xfhpc.sendMat(C, idxKernel, idxDevice)
        xfhpc.sendMat(X, idxKernel, idxDevice)
        xfhpc.fcnOp(
            A,
            B,
            C,
            X,
            post_val[0],
            post_val[1],
            pRelu_val[0],
            pRelu_val[1],
            idxKernel,
            idxDevice)
        xfhpc.getMat(C, idxKernel, idxDevice)
        self.cmpWithinTolerance(C, golden_C)
