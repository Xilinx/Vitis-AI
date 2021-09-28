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


class XFBLASManager:
    def __init__(self, libFile):
        self._lib = cdll.LoadLibrary(libFile)
        self._lib.xfblasCreate.argtypes = [c_char_p, c_char_p, c_uint, c_uint]
        self._lib.xfblasCreate.restype = c_bool
        self._lib.xfblasSend.argtypes = [
            np.ctypeslib.ndpointer(
                flags="C_CONTIGUOUS"),
            c_ulonglong,
            c_uint,
            c_uint,
            c_uint]
        self._lib.xfblasSend.restype = c_bool
        self._lib.xfblasGet.argtypes = [
            np.ctypeslib.ndpointer(
                flags="C_CONTIGUOUS"), c_uint, c_uint]
        self._lib.xfblasGet.restype = c_bool
        self._lib.xfblasFreeInstr.argtypes = [c_uint, c_uint]
        self._lib.xfblasDestroy.argtypes = [c_uint, c_uint]
        self._lib.xfblasFree.argtypes = [
            np.ctypeslib.ndpointer(
                flags="C_CONTIGUOUS"), c_uint, c_uint]
        self._lib.xfblasGemm.argtypes = [
            c_uint, c_uint, c_uint, c_uint, np.ctypeslib.ndpointer(
                flags="C_CONTIGUOUS"), c_uint, np.ctypeslib.ndpointer(
                flags="C_CONTIGUOUS"), c_uint, c_uint, np.ctypeslib.ndpointer(
                flags="C_CONTIGUOUS"), c_uint, c_uint, c_uint]
        self._lib.xfblasGemm.restype = c_bool
        self._lib.xfblasGemv.argtypes = [
            c_uint, c_uint, c_uint, np.ctypeslib.ndpointer(
                flags="C_CONTIGUOUS"), c_uint, np.ctypeslib.ndpointer(
                flags="C_CONTIGUOUS"), c_uint, c_uint, np.ctypeslib.ndpointer(
                flags="C_CONTIGUOUS"), c_uint, c_uint, c_uint]
        self._lib.xfblasGemv.restype = c_bool
        self._lib.xfblasGemm.restype = c_bool
        self._lib.xfblasGetByAddress.argtypes = [np.ctypeslib.ndpointer(
            flags="C_CONTIGUOUS"), c_ulonglong, c_uint, c_uint, c_uint]
        self._lib.xfblasGetByAddress.restype = c_bool
        self._lib.xfblasExecuteAsync.argtypes = [c_uint, c_uint]
        self._lib.xfblasExecute.argtypes = [c_uint, c_uint]

    def createGemm(self, xclbin, numKernel, idxDevice):
        '''
        create Gemm Handle


        Parameters

        xclbin
                    file path for FPGA bitstream
        numKernel
                    number of CUs in the xclbin
        idxDeivce
                    index of local device to be used
        '''
        b_xclbin = xclbin.encode('utf-8')
        b_log = xclbin.encode('utf-8')
        return self._lib.xfblasCreate(b_xclbin, b'Gemm', numKernel, idxDevice)

    def createGemv(self, xclbin, numKernel, idxDevice):
        b_xclbin = xclbin.encode('utf-8')
        b_log = xclbin.encode('utf-8')
        return self._lib.xfblasCreate(b_xclbin, b'Gemv', numKernel, idxDevice)

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
        return self._lib.xfblasSend(
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
        return self._lib.xfblasGet(A, idxKernel, idxDevice)

    def freeInstr(self, idxKernel, idxDevice):
        '''
        free memory for instructions

        Parameters

        idxKernel
                    index of kernel to be used
        idxDeivce
                    index of local device to be used
        '''
        return self._lib.xfblasFreeInstr(idxKernel, idxDevice)

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
        return self._lib.xfblasFree(A, idxKernel, idxDevice)

    def destroy(self, numKernel, idxDevice):
        '''
        release handle used by the XFBLAS library

        Parameters

        numKernel
                    number of CUs in the xclbin
        idxDeivce
                    index of local device to be used
        '''
        return self._lib.xfblasDestroy(numKernel, idxDevice)

    def gemmOp(self, A, B, C, idxKernel, idxDevice):
        '''
        perform matrix-matrix multiplication of C=A*B

        Parameters

        A:              ndarray
                        matrix in host memory
        B:              ndarray
                        matrix in host memory
        C:              ndarray
                        matrix in host memory
        idxKernel:      int
                        index of kernel to be used
        idxDeivce:      int
                        index of local device to be used
        '''
        return self._lib.xfblasGemm(
            c_uint(
                A.shape[0]), c_uint(
                B.shape[1]), c_uint(
                A.shape[1]), 1, A, c_uint(
                    A.shape[1]), B, c_uint(
                        B.shape[1]), 1, C, c_uint(
                            B.shape[1]), idxKernel, idxDevice)

    def gemvOp(self, A, x, y, idxKernel, idxDevice):
        return self._lib.xfblasGemv(
            c_uint(
                A.shape[0]), c_uint(
                A.shape[1]), 1, A, c_uint(
                A.shape[1]), x, 1, y, 1, idxKernel, idxDevice)
                
    def getMatByAddress(self, A, offset, idxKernel, idxDevice):
        return self._lib.xfblasGetByAddress(A, c_ulonglong(
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
        return self._lib.xfblasExecuteAsync(numKernel, idxDevice)

    def execute(self, idxKernel, idxDevice):
        '''
        run ith kernel

        Parameters

        idxKernel:      int
                        index of kernel to be used
        idxDeivce:      int
                        index of local device to be used
        '''
        return self._lib.xfblasExecute(idxKernel, idxDevice)


_xfblasManager = None


def createGemm(args, xclbin_opts, numKernel=1, idxDevice=0):
    if int(xclbin_opts['BLAS_runGemm']) != 1:
        raise Exception('The xclbin does not include gemm engine.')
    createManager(args.lib)
    return _xfblasManager.createGemm(args.xclbin, numKernel, idxDevice)


def createGemv(args, xclbin_opts, numKernel=1, idxDevice=0):
    if int(xclbin_opts['BLAS_runGemv']) != 1:
        raise Exception('The xclbin does not include gemv engine.')
    createManager(args.lib)
    return _xfblasManager.createGemv(args.xclbin, numKernel, idxDevice)


def sendMat(A, idxKernel=0, idxDevice=0):
    return _xfblasManager.sendMat(A, idxKernel, idxDevice)


def getMat(A, idxKernel=0, idxDevice=0):
    return _xfblasManager.getMat(A, idxKernel, idxDevice)


def freeInstr(idxKernel=0, idxDevice=0):
    return _xfblasManager.freeInstr(idxKernel, idxDevice)


def freeMat(A, idxKernel=0, idxDevice=0):
    return _xfblasManager.freeMat(A, idxKernel, idxDevice)


def destroy(numKernel=1, idxDevice=0):
    return _xfblasManager.destroy(numKernel, idxDevice)


def gemmOp(A, B, C, idxKernel=0, idxDevice=0):
    return _xfblasManager.gemmOp(A, B, C, idxKernel, idxDevice)


def gemvOp(A, x, y, idxKernel=0, idxDevice=0):
    return _xfblasManager.gemvOp(A, x, y, idxKernel, idxDevice)

def getMatByAddress(A, offset, idxKernel=0, idxDevice=0):
    return _xfblasManager.getMatByAddress(A, offset, idxKernel, idxDevice)


def executeAsync(numKernel=1, idxDevice=0):
    return _xfblasManager.executeAsync(numKernel, idxDevice)


def execute(idxKernel=0, idxDevice=0):
    return _xfblasManager.execute(idxKernel, idxDevice)


def createManager(libFile):
    global _xfblasManager
    if not _xfblasManager:
        _xfblasManager = XFBLASManager(libFile)
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
    parser = argparse.ArgumentParser(description='xfblas')
    parser.add_argument(
        '--xclbin',
        required=True,
        help='file path to FPGA bitstream')
    parser.add_argument(
        '--lib',
        required=True,
        help='file path to xfblas shared library')
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
