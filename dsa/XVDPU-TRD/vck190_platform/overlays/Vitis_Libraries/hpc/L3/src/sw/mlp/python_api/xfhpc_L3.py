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

from ctypes import *
import numpy as np
import sys
import argparse
import os


class XFHPCManager:
    def __init__(self, libFile):
        self._lib = cdll.LoadLibrary(libFile)
        self._lib.xfhpcCreate.argtypes = [c_char_p, c_uint, c_uint]
        self._lib.xfhpcCreate.restype = c_bool
        self._lib.xfhpcSend.argtypes = [
            np.ctypeslib.ndpointer(
                flags="C_CONTIGUOUS"),
            c_ulonglong,
            c_uint,
            c_uint,
            c_uint]
        self._lib.xfhpcSend.restype = c_bool
        self._lib.xfhpcGet.argtypes = [
            np.ctypeslib.ndpointer(
                flags="C_CONTIGUOUS"), c_uint, c_uint]
        self._lib.xfhpcGet.restype = c_bool
        self._lib.xfhpcFreeInstr.argtypes = [c_uint, c_uint]
        self._lib.xfhpcDestroy.argtypes = [c_uint, c_uint]
        self._lib.xfhpcFree.argtypes = [
            np.ctypeslib.ndpointer(
                flags="C_CONTIGUOUS"), c_uint, c_uint]
        self._lib.xfhpcFcn.argtypes = [c_uint, c_uint, c_uint, c_uint,
                                       np.ctypeslib.ndpointer(flags="C_CONTIGUOUS"), c_uint,
                                       np.ctypeslib.ndpointer(flags="C_CONTIGUOUS"), c_uint,
                                       c_uint,
                                       np.ctypeslib.ndpointer(flags="C_CONTIGUOUS"), c_uint,
                                       np.ctypeslib.ndpointer(flags="C_CONTIGUOUS"), c_uint,
                                       c_int, c_int,
                                       c_short, c_short,
                                       c_uint, c_uint]
        self._lib.xfhpcFcnByAddress.argtypes = [
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_int,
            c_int,
            c_short,
            c_short,
            c_uint,
            c_uint]
        self._lib.xfhpcGetByAddress.argtypes = [np.ctypeslib.ndpointer(
            flags="C_CONTIGUOUS"), c_ulonglong, c_uint, c_uint, c_uint]
        self._lib.xfhpcGetByAddress.restype = c_bool
        self._lib.xfhpcExecuteAsync.argtypes = [c_uint, c_uint]
        self._lib.xfhpcExecute.argtypes = [c_uint, c_uint]

    def createFcn(self, xclbin, numKernel, idxDevice):
        b_xclbin = xclbin.encode('utf-8')
        return self._lib.xfhpcCreate(b_xclbin, numKernel, idxDevice)

    def sendMat(self, A, idxKernel, idxDevice):
        '''
        send mat from host to device

        Parameters

        A:          ndarray
                    matrix in host memory
        idxKernel:  int
                    index of kernel to be used
        idxDeivce:  int
                    index of local device to be used
        '''
        return self._lib.xfhpcSend(
            A, c_ulonglong(
                A.size), c_uint(
                A.itemsize), idxKernel, idxDevice)

    def getMat(self, A, idxKernel, idxDevice):
        '''
        get mat from device to host

        Parameters

        A:          ndarray
                    matrix in host memory
        idxKernel:  int
                    index of kernel to be used
        idxDeivce:  int
                    index of local device to be used
        '''
        return self._lib.xfhpcGet(A, idxKernel, idxDevice)

    def freeInstr(self, idxKernel, idxDevice):
        '''
        free memory for instructions

        Parameters

        idxKernel
                    index of kernel to be used
        idxDeivce
                    index of local device to be used
        '''
        return self._lib.xfhpcFreeInstr(idxKernel, idxDevice)

    def freeMat(self, A, idxKernel, idxDevice):
        '''
        free device memory for mat A

        Parameters

        A:          ndarray
                    matrix in host memory
        idxKernel:  int
                    index of kernel to be used
        idxDeivce:  int
                    index of local device to be used
        '''
        return self._lib.xfhpcFree(A, idxKernel, idxDevice)

    def destroy(self, numKernel, idxDevice):
        '''
        release handle used by the XFHPC library

        Parameters

        numKernel
                    number of CUs in the xclbin
        idxDeivce
                    index of local device to be used
        '''
        return self._lib.xfhpcDestroy(numKernel, idxDevice)

    def fcnOp(
            self,
            A,
            B,
            C,
            X,
            postScale,
            postShift,
            preluScale,
            preluAlpha,
            idxKernel,
            idxDevice):
        return self._lib.xfhpcFcn(
            c_uint(
                A.shape[0]), c_uint(
                B.shape[1]), c_uint(
                A.shape[1]), 1, A, c_uint(
                    A.shape[1]), B, c_uint(
                        B.shape[1]), 1, C, c_uint(
                            C.shape[1]), X, c_uint(
                                X.shape[1]), postScale, postShift, preluScale, preluAlpha, idxKernel, idxDevice)

    def fcnOpByAddress(
        self,
        a,
        b,
        c,
        x,
        A,
        B,
        C,
        X,
        postScale,
        postShift,
        preluScale,
        preluAlpha,
        idxKernel,
            idxDevice):
        return self._lib.xfhpcFcnByAddress(
            c_uint(a), c_uint(b), c_uint(c), c_uint(x), c_uint(
                A.shape[0]), c_uint(
                B.shape[1]), c_uint(
                A.shape[1]), c_uint(
                    A.shape[1]), c_uint(
                        B.shape[1]), c_uint(
                            C.shape[1]), c_uint(
                                X.shape[1]), postScale, postShift, preluScale, preluAlpha, idxKernel, idxDevice)

    def getMatByAddress(self, A, offset, idxKernel, idxDevice):
        return self._lib.xfhpcGetByAddress(A, c_ulonglong(
            A.size * A.itemsize), offset, idxKernel, idxDevice)

    def executeAsync(self, numKernel, idxDevice):
        '''
        run number of kernels async

        Parameters

        numKernel
                    number of CUs in the xclbin
        idxDeivce
                    index of local device to be used
        '''
        return self._lib.xfhpcExecuteAsync(numKernel, idxDevice)

    def execute(self, idxKernel, idxDevice):
        '''
        run ith kernel

        Parameters

        idxKernel:      int
                        index of kernel to be used
        idxDeivce:      int
                        index of local device to be used
        '''
        return self._lib.xfhpcExecute(idxKernel, idxDevice)


_xfhpcManager = None


def createFcn(args, xclbin_opts, numKernel=1, idxDevice=0):
    if int(xclbin_opts['BLAS_runFcn']) != 1:
        raise Exception('The xclbin does not include fcn engine.')
    createManager(args.lib)
    return _xfhpcManager.createFcn(args.xclbin, numKernel, idxDevice)


def sendMat(A, idxKernel=0, idxDevice=0):
    return _xfhpcManager.sendMat(A, idxKernel, idxDevice)


def getMat(A, idxKernel=0, idxDevice=0):
    return _xfhpcManager.getMat(A, idxKernel, idxDevice)


def freeInstr(idxKernel=0, idxDevice=0):
    return _xfhpcManager.freeInstr(idxKernel, idxDevice)


def freeMat(A, idxKernel=0, idxDevice=0):
    return _xfhpcManager.freeMat(A, idxKernel, idxDevice)


def destroy(numKernel=1, idxDevice=0):
    return _xfhpcManager.destroy(numKernel, idxDevice)


def fcnOp(
        A,
        B,
        C,
        X,
        postScale=1,
        postShift=0,
        preluScale=1,
        preluAlpha=0,
        idxKernel=0,
        idxDevice=0):
    return _xfhpcManager.fcnOp(
        A,
        B,
        C,
        X,
        postScale,
        postShift,
        preluScale,
        preluAlpha,
        idxKernel,
        idxDevice)


def fcnOpByAddress(
    a,
    b,
    c,
    x,
    A,
    B,
    C,
    X,
    postScale=1,
    postShift=0,
    preluScale=1,
    preluAlpha=0,
    idxKernel=0,
        idxDevice=0):
    return _xfhpcManager.fcnOpByAddress(
        a,
        b,
        c,
        x,
        A,
        B,
        C,
        X,
        postScale,
        postShift,
        preluScale,
        preluAlpha,
        idxKernel,
        idxDevice)


def getMatByAddress(A, offset, idxKernel=0, idxDevice=0):
    return _xfhpcManager.getMatByAddress(A, offset, idxKernel, idxDevice)


def executeAsync(numKernel=1, idxDevice=0):
    return _xfhpcManager.executeAsync(numKernel, idxDevice)


def execute(idxKernel=0, idxDevice=0):
    return _xfhpcManager.execute(idxKernel, idxDevice)


def createManager(libFile):
    global _xfhpcManager
    if not _xfhpcManager:
        _xfhpcManager = XFHPCManager(libFile)
    return True


def parse_cfg(filename):
    myvars = {}
    with open(filename) as myfile:
        for line in myfile:
            for word in line.split():
                name, var = word.split("=")
                myvars[name.strip()] = var.rstrip()
    return myvars


def default_args():
    parser = argparse.ArgumentParser(description='xfhpc')
    parser.add_argument(
        '--xclbin',
        required=True,
        help='file path to FPGA bitstream')
    parser.add_argument(
        '--lib',
        required=True,
        help='file path to xfhpc shared library')
    parser.add_argument(
        '--cfg',
        required=True,
        help='file describing .xclbin properties')
    return parser


def processCommandLine():
    parser = default_args()
    args = parser.parse_args()
    xclbin_opts = parse_cfg(args.cfg)
    return args, xclbin_opts
