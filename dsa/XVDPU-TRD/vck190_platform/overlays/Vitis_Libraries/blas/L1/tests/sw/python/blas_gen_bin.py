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

import pdb
import ctypes as ct
import numpy as np
import argparse
import os


class BLAS_ERROR(Exception):
    def __init__(self, status, message):
        self.status = status
        self.message = message


class BLAS_GEN:

    def __init__(self, lib):
        c_types = [ct.c_byte, ct.c_short, ct.c_int, ct.c_long, ct.c_longlong,
                   ct.c_ubyte, ct.c_ushort, ct.c_uint, ct.c_ulong,
                   ct.c_ulonglong, ct.c_float, ct.c_double]
        self.lib = lib
        self.lib.genBinNew.restype = ct.c_void_p
        self.obj = lib.genBinNew()
        self.typeDict = {np.dtype(ctype): ctype for ctype in c_types}
        self.status = ['XFBLAS_STATUS_SUCCESS',  # 0
                       'XFBLAS_STATUS_NOT_INITIALIZED',  # 1
                       'XFBLAS_STATUS_INVALID_VALUE',  # 2
                       'XFBLAS_STATUS_ALLOC_FAILED',  # 3
                       'XFBLAS_STATUS_NOT_SUPPORTED',  # 4
                       'XFBLAS_STATUS_NOT_PADDED',  # 5
                       'XFBLAS_STATUS_MEM_ALLOCATED',  # 6
                       'XFBLAS_STATUS_INVALID_OP',  # 7
                       'XFBLAS_STATUS_INVALID_FILE',  # 8
                       'XFBLAS_STATUS_INVALID_PROGRAM']  # 9

    def _getType(self, x):
        return self.typeDict[x.dtype]

    def _getPointer(self, x):
        try:
            ptr_x = ct.pointer(np.ctypeslib.as_ctypes(x))
            return ptr_x
        except BaseException:
            return None

    def addB1Instr(
            self,
            p_opName,
            p_n,
            p_alpha,
            p_x,
            p_y,
            p_xRes,
            p_yRes,
            p_res):
        func = self.lib.addB1Instr
        func.argtypes = [
            ct.c_void_p,
            ct.c_char_p,
            ct.c_int,
            self._getType(p_alpha),
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            self._getType(p_res)]
        status = func(
            self.obj,
            p_opName.encode('utf-8'),
            p_n,
            p_alpha,
            self._getPointer(p_x),
            self._getPointer(p_y),
            self._getPointer(p_xRes),
            self._getPointer(p_yRes),
            p_res)
        if status > 0:
            raise BLAS_ERROR(
                self.status[status],
                "Add BLAS_L1 instruction failed.")

    def addB2Instr(self, p_opName, p_m, p_n, p_kl, p_ku,
                   p_alpha, p_beta, p_a, p_x, p_y, p_aRes, p_yRes):
        func = self.lib.addB2Instr
        func.argtypes = [
            ct.c_void_p,
            ct.c_char_p,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            self._getType(p_alpha),
            self._getType(p_beta),
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p]
        status = func(
            self.obj,
            p_opName.encode('utf-8'),
            p_m,
            p_n,
            p_kl,
            p_ku,
            p_alpha,
            p_beta,
            self._getPointer(p_a),
            self._getPointer(p_x),
            self._getPointer(p_y),
            self._getPointer(p_aRes),
            self._getPointer(p_yRes))
        if status > 0:
            raise BLAS_ERROR(
                self.status[status],
                "Add BLAS_L2 instruction failed.")

    def write2BinFile(self, p_fileName):
        func = self.lib.write2BinFile
        func.argtypes = [ct.c_void_p, ct.c_char_p]
        status = func(self.obj, p_fileName.encode('utf-8'))
        if status > 0:
            raise BLAS_ERROR(
                self.status[status],
                "Write file %s failed." %
                p_fileName)

    def readFromBinFile(self, p_fileName):
        func = self.lib.readFromBinFile
        func.argtypes = [ct.c_void_p, ct.c_char_p]
        status = func(self.obj, p_fileName.encode('utf-8'))
        if status > 0:
            raise BLAS_ERROR(
                self.status[status],
                "Read file %s failed." %
                p_fileName)

    def printProgram(self):
        func = self.lib.printProgram
        func.argtypes = [ct.c_void_p]
        func(self.obj)

    def __del__(self):
        func = self.lib.genBinDel
        func.argtypes = [ct.c_void_p]
        func(self.obj)


def main(lib, path):
    blas_read = BLAS_GEN(lib)
    blas_read.readFromBinFile(path)
    blas_read.printProgram()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run HLS test.')
    parser.add_argument(
        'so',
        type=str,
        metavar='sharedLibrary',
        help='path to the shared library file')
    parser.add_argument('bin', type=str, metavar='binfile',
                        help='path to generate bin files')
    args = parser.parse_args()
    lib = ct.cdll.LoadLibrary(args.so)
    main(lib, args.bin)
