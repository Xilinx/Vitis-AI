'''
Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from ctypes import *
import numpy as np
try:
    pyc_libn2cube = cdll.LoadLibrary("libn2cube.so")
except Exception:
    print('Load libn2cube.so failed\nPlease install DNNDK first!')
pyc_libn2cube.pyc_dpuLoadKernel.restype = POINTER(c_void_p)
pyc_libn2cube.pyc_dpuCreateTask.restype = POINTER(c_void_p)
pyc_libn2cube.pyc_dpuGetTaskProfile.restype = c_longlong
pyc_libn2cube.pyc_dpuGetNodeProfile.restype = c_longlong
pyc_libn2cube.pyc_dpuGetInputTensor.restype = POINTER(c_void_p)
pyc_libn2cube.pyc_dpuGetInputTensorScale.restype = c_float
pyc_libn2cube.pyc_dpuGetOutputTensor.restype = POINTER(c_void_p)
pyc_libn2cube.pyc_dpuGetOutputTensorAddress.restype = POINTER(c_byte)
pyc_libn2cube.pyc_dpuGetOutputTensorScale.restype = c_float
pyc_libn2cube.pyc_dpuGetTensorScale.restype = c_float
pyc_libn2cube.pyc_dpuGetTensorAddress.restype = POINTER(c_byte)
pyc_libn2cube.pyc_dpuSetInputTensorInCHWInt8.argtypes = (
    POINTER(c_void_p), c_char_p, np.ctypeslib.ndpointer(c_byte), c_int, c_int)
pyc_libn2cube.pyc_dpuSetInputTensorInCHWFP32.argtypes = (
    POINTER(c_void_p), c_char_p, np.ctypeslib.ndpointer(c_float), c_int, c_int)
pyc_libn2cube.pyc_dpuSetInputTensorInHWCInt8.argtypes = (
    POINTER(c_void_p), c_char_p, np.ctypeslib.ndpointer(c_byte), c_int, c_int)
pyc_libn2cube.pyc_dpuSetInputTensorInHWCFP32.argtypes = (
    POINTER(c_void_p), c_char_p,  np.ctypeslib.ndpointer(c_float), c_int, c_int)
pyc_libn2cube.pyc_dpuGetOutputTensorInCHWInt8.argtypes = (
    POINTER(c_void_p), c_char_p, np.ctypeslib.ndpointer(c_byte), c_int, c_int)
pyc_libn2cube.pyc_dpuGetOutputTensorInCHWFP32.argtypes = (
    POINTER(c_void_p), c_char_p, np.ctypeslib.ndpointer(c_float), c_int, c_int)
pyc_libn2cube.pyc_dpuGetOutputTensorInHWCInt8.argtypes = (
    POINTER(c_void_p), c_char_p, np.ctypeslib.ndpointer(c_byte), c_int, c_int)
pyc_libn2cube.pyc_dpuGetOutputTensorInHWCFP32.argtypes = (
    POINTER(c_void_p), c_char_p, np.ctypeslib.ndpointer(c_float), c_int, c_int)
pyc_libn2cube.pyc_dpuRunSoftmax.argtypes = (
    POINTER(c_byte), np.ctypeslib.ndpointer(c_float), c_int, c_int, c_float)
pyc_libn2cube.dpuGetExceptionMessage.restype = POINTER(c_char)
pyc_libn2cube.pyc_dpuGetBoundaryIOTensor.restype = POINTER(c_void_p)
def dpuOpen():
    """
    Open & initialize the usage of DPU device
    Returns: 0 on success, or negative value in case of failure.
             Error message (Fail to open DPU device) is reported if any error takes place
    """
    return pyc_libn2cube.pyc_dpuOpen()


def dpuClose():
    """
    Close & finalize the usage of DPU devicei
    Returns: 0 on success, or negative error ID in case of failure.
             Error message (Fail to close DPU device) is reported if any error takes place
    """
    return pyc_libn2cube.pyc_dpuClose()


def dpuLoadKernel(kernelName):
    """
    Load a DPU Kernel and allocate DPU memory space for
    its Code/Weight/Bias segments
    kernelName: The pointer to neural network name.
                Use the names produced by Deep Neural Network Compiler (DNNC) after
                the compilation of neural network.
                For each DL application, perhaps there are many DPU Kernels existing
                in its hybrid CPU+DPU binary executable. For each DPU Kernel, it has
                one unique name for differentiation purpose
    Returns: The loaded DPU Kernel on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuLoadKernel(c_char_p(kernelName.encode("utf-8")))


def dpuDestroyKernel(kernel):
    """
    Destroy a DPU Kernel and release its associated resources
    kernel:  The DPU Kernel to be destroyed. This parameter should be gotten from the result of dpuLoadKernel()
    Returns: 0 on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuDestroyKernel(kernel)


def dpuCreateTask(kernel, mode):
    """
    Instantiate a DPU Task from one DPU Kernel, allocate its private
    working memory buffer and prepare for its execution context
    kernel:  The DPU Kernel. This parameter should be gotten from the result of dpuLoadKernel()
    mode:    The running mode of DPU Task. There are 3 available modes:
               MODE_NORMAL: default mode identical to the mode value 0.
               MODE_PROF: output profiling information layer by layer while running of DPU Task,
                     which is useful for performance analysis.
               MODE_DUMP: dump the raw data for DPU Task's CODE/BIAS/WEIGHT/INPUT/OUTPUT layer by layer
    Returns: 0 on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuCreateTask(kernel, c_int(mode))


def dpuDestroyTask(task):
    """
    Remove a DPU Task, release its working memory buffer and destroy
    associated execution context
    task:    DPU Task. This parameter should be gotten from the result of dpuCreatTask()
    Returns: 0 on success, or negative value in case of any failure
    """
    return pyc_libn2cube.pyc_dpuDestroyTask(task)


def dpuRunTask(task):
    """
    Launch the running of DPU Task
    task:    DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    Returns: 0 on success, or negative value in case of any failure
    """
    return pyc_libn2cube.pyc_dpuRunTask(task)


def dpuEnableTaskDebug(task):
    """
    Enable dump facility of DPU Task while running for debugging purpose
    task:    DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    Returns: 0 on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuEnableTaskDebug(task)


def dpuEnableTaskProfile(task):
    """
    Enable profiling facility of DPU Task while running to get its performance metrics
    task:    DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    Returns: 0 on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuEnableTaskProfile(task)


def dpuGetTaskProfile(task):
    """
    Get the execution time of DPU Task
    task:    DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    Returns: The DPU Task's execution time (us) after its running
    """
    return pyc_libn2cube.pyc_dpuGetTaskProfile(task)


def dpuGetNodeProfile(task, nodeName):
    """
    Get the execution time of DPU Node
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    Returns:  The DPU Node's execution time (us) after its running
    """
    return pyc_libn2cube.pyc_dpuGetNodeProfile(task, c_char_p(nodeName.encode("utf-8")))


"""
API for both single IO and multiple IO.
For multiply IO, should specify the input/output tensor idx.
"""


def dpuGetInputTensorCnt(task, nodeName):
    """
    Get total number of input Tensor of DPU Task
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The total number of input tensor for specified Node.
    """
    return pyc_libn2cube.pyc_dpuGetInputTensorCnt(task, c_char_p(nodeName.encode("utf-8")))


def dpuGetInputTensor(task, nodeName, idx=0):
    """
    Get input Tensor of DPU Task
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The pointer to Task input Tensor on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetInputTensor(task,
                                               c_char_p(nodeName.encode("utf-8")), c_int(idx))

def dpuGetTensorData(tensorAddress,  tensorSize):
    """
    Get the tensor data from the address that returnd by dpuGetOutputTensorAddress
    tensorAddress: Result from dpuGetOutputTensorAddress()
    tensorSize:    Size of the output data
    Returns:       Data output.
    """
    a = np.fromiter(tensorAddress, dtype=np.int8,count=tensorSize)
    return a

#def dpuGetTensorData(tensorAddress, data, tensorSize):
#    """
#    Get the tensor data from the address that returnd by dpuGetOutputTensorAddress
#    tensorAddress: Result from dpuGetOutputTensorAddress()
#    data:          Output data
#    tensorSize:    Size of the output data
#    Returns:       -
#    """
#    for i in range(tensorSize):
#        data[i] = int(tensorAddress[i])
#    return


def dpuGetInputTensorSize(task, nodeName, idx=0):
    """
    Get the size (in byte) of one DPU Task input Tensor
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The size of Task's input Tensor on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetInputTensorSize(
        task, c_char_p(nodeName.encode("utf-8")), c_int(idx))


def dpuGetInputTensorScale(task, nodeName, idx=0):
    """
    Get the scale value (DPU INT8 quantization) of one DPU Task input Tensor
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Retruns:  The scale value of Task's input Tensor on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetInputTensorScale(task,
                                                    c_char_p(nodeName.encode("utf-8")),
                                                    c_int(idx))


def dpuGetInputTensorHeight(task, nodeName, idx=0):
    """
    Get the height dimension of one DPU Task input Tensor
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The height dimension of Task's input Tensor on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetInputTensorHeight(task,
                                                     c_char_p(nodeName.encode("utf-8")),
                                                     c_int(idx))


def dpuGetInputTensorWidth(task, nodeName, idx=0):
    """
    Get the width dimension of one DPU Task input Tensor
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The width dimension of Task's input Tensor on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetInputTensorWidth(task,
                                                    c_char_p(nodeName.encode("utf-8")),
                                                    c_int(idx))


def dpuGetInputTensorChannel(task, nodeName, idx=0):
    """
    Get the channel dimension of one DPU Task input Tensor
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The channel dimension of Task's input Tensor on success, or report error in case of any failure.
    """
    return pyc_libn2cube.pyc_dpuGetInputTensorChannel(task,
                                                      c_char_p(nodeName.encode("utf-8")),
                                                      c_int(idx))


def dpuGetOutputTensorCnt(task, nodeName):
    """
    Get total number of output Tensor of DPU Task
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    Returns:  The total number of output tensor for the DPU Task
    """
    return pyc_libn2cube.pyc_dpuGetOutputTensorCnt(task, c_char_p(nodeName.encode("utf-8")))


def dpuGetOutputTensor(task, nodeName, idx=0):
    """
    Get output Tensor of one DPU Task
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The pointer to Task's output Tensor on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetOutputTensor(task,
                                                c_char_p(nodeName.encode("utf-8")), c_int(idx))


def dpuGetOutputTensorSize(task, nodeName, idx=0):
    """
    Get the size (in byte) of one DPU Task output Tensor
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The size of Task's output Tensor on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetOutputTensorSize(task,
                                                    c_char_p(nodeName.encode("utf-8")),
                                                    c_int(idx))


def dpuGetOutputTensorAddress(task, nodeName, idx=0):
    """
    Get the start address of one DPU Task output Tensor
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The start addresses to Task's output Tensor on success, or report error in case of any failure 
    """
    size = dpuGetOutputTensorSize(task, nodeName, idx)
    output = (c_int8*size)()
    outputPP = POINTER(c_int8)(output)
    outputPP = pyc_libn2cube.pyc_dpuGetOutputTensorAddress(task,
                                                           c_char_p(nodeName.encode("utf-8")),
                                                           c_int(idx))
    return outputPP


def dpuGetOutputTensorScale(task, nodeName, idx=0):
    """
    Get the scale value (DPU INT8 quantization) of one DPU Task output Tensor
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The scale value of Task's output Tensor on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetOutputTensorScale(task,
                                                     c_char_p(nodeName.encode("utf-8")),
                                                     c_int(idx))


def dpuGetOutputTensorHeight(task, nodeName, idx=0):
    """
    Get the height dimension of one DPU Task output Tensor
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The height dimension of Task's output Tensor on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetOutputTensorHeight(task,
                                                      c_char_p(nodeName.encode("utf-8")),
                                                      c_int(idx))


def dpuGetOutputTensorWidth(task, nodeName, idx=0):
    """
    Get the channel dimension of one DPU Task output Tensor
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The width dimension of Task's output Tensor on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetOutputTensorWidth(task,
                                                     c_char_p(nodeName.encode("utf-8")),
                                                     c_int(idx))


def dpuGetOutputTensorChannel(task, nodeName, idx=0):
    """
    Get DPU Node's output tensor's channel
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The channel dimension of Task's output Tensor on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetOutputTensorChannel(task,
                                                       c_char_p(nodeName.encode("utf-8")),
                                                       c_int(idx))


def dpuGetTensorSize(tensor):
    """
    Get the size of one DPU Tensor
    tensor:  DPU tensor. This parameter should be gotten from the reslut of dpuGetOutputTensor()
    Returns: The size of Tensor, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetTensorSize(tensor)


def dpuGetTensorAddress(tensor):
    """
    Get the address of one DPU tensor
    tensor:  DPU tensor. This parameter should be gotten from the reslut of dpuGetOutputTensor()
    Returns: The pointer of Tensor list, or report error in case of any failure
    """
    size = dpuGetTensorSize(tensor)
    output = (c_int8*size)()
    outputPP = POINTER(c_int8)(output)
    outputPP = pyc_libn2cube.pyc_dpuGetTensorAddress(tensor)
    return outputPP


def dpuGetTensorScale(tensor):
    """
    Get the scale value of one DPU Tensor
    Returns: The scale value of Tensor, or report error in case of any failure
    tensor:  DPU tensor. This parameter should be gotten from the reslut of dpuGetOutputTensor()
    Returns: The scale value of Tensor, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetTensorScale(tensor)


def dpuGetTensorHeight(tensor):
    """
    Get the height dimension of one DPU Tensor
    Returns: The height dimension of Tensor, or report error in case of any failure
    tensor:  DPU tensor. This parameter should be gotten from the reslut of dpuGetOutputTensor()
    Returns: The height dimension of Tensor, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetTensorHeight(tensor)


def dpuGetTensorWidth(tensor):
    """
    Get the width dimension of one DPU Tensor
    Returns: The width dimension of Tensor, or report error in case of any failure
    tensor:  DPU tensor. This parameter should be gotten from the reslut of dpuGetOutputTensor()
    Returns: The width dimension of Tensor, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetTensorWidth(tensor)


def dpuGetTensorChannel(tensor):
    """
    Get the channel dimension of one DPU Tensor
    Returns: The channel dimension of Tensor, or report error in case of any failure
    tensor:  DPU tensor. This parameter should be gotten from the reslut of dpuGetOutputTensor()
    Returns: The channel dimension of Tensor, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetTensorChannel(tensor)


def dpuSetInputTensorInCHWInt8(task, nodeName, data, size, idx=0):
    """
    Set DPU Task's input Tensor with data stored under Caffe
    Blob's order (channel/height/width) in INT8 format 
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    data:     The input data
    size:     The size (in Bytes) of input data to be stored
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  0 on success, or report error in case of failure
    """
    try:
        rtn = pyc_libn2cube.pyc_dpuSetInputTensorInCHWInt8(task,
                                                        c_char_p(nodeName.encode("utf-8")),
                                                        data,
                                                        c_int(size), c_int(idx))
    except Exception:
        print('Please input data as dtype=np.int8')
        return -1
    return rtn

def dpuSetInputTensorInCHWFP32(task, nodeName, data, size, idx=0):
    """
    Set DPU Task's input Tensor with data stored under Caffe
    Blob's order (channel/height/width) in FP32 format
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    data:     The input data
    size:     The size (in Bytes) of input data to be stored
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  0 on success, or report error in case of failure
    """
    try:
        rtn = pyc_libn2cube.pyc_dpuSetInputTensorInCHWFP32(task,
                                                        c_char_p(nodeName.encode("utf-8")),
                                                        data,
                                                        c_int(size), c_int(idx))
    except Exception:
        print('Please input data as dtype=np.float32')
        return -1
    return rtn


def dpuSetInputTensorInHWCInt8(task, nodeName, data, size, idx=0):
    """
    Set DPU Task's input Tensor with data stored under DPU
    Tensor's order (height/width/channel) in INT8 format
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    data:     The input data
    size:     The size (in Bytes) of input data to be stored
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  0 on success, or report error in case of failure
    """
    try:
        rtn = pyc_libn2cube.pyc_dpuSetInputTensorInHWCInt8(task,
                                                        c_char_p(nodeName.encode("utf-8")),
                                                        data,
                                                        c_int(size), c_int(idx))
    except Exception:
        print('Please input data as dtype=np.int8')
        return -1
    return rtn


def dpuSetInputTensorInHWCFP32(task, nodeName, data, size, idx=0):
    """
    Set DPU Task's input Tensor with data stored under DPU
    Tensor's order (height/width/channel) in FP32 format
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    data:     The input data
    size:     The size (in Bytes) of input data to be stored
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  0 on success, or report error in case of failure
    """
    
    try:
        rtn = pyc_libn2cube.pyc_dpuSetInputTensorInHWCFP32(task,
                                                        c_char_p(nodeName.encode("utf-8")),
                                                        data,
                                                        c_int(size), c_int(idx))
    except Exception:
        print('Please input data as dtype=np.float32')
        return -1
    return rtn



def dpuGetOutputTensorInCHWInt8(task, nodeName, size, idx=0):
    """
    Get DPU Task's output Tensor and store them under DPU
    Tensor's order (height/width/channel) in INT8 format
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    size:     The size (in Bytes) of output data to be stored
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The output Tensor's data
    """
    output = np.zeros(size, dtype = np.int8)
    rtn = pyc_libn2cube.pyc_dpuGetOutputTensorInCHWInt8(task,
                                                        c_char_p(nodeName.encode("utf-8")),
                                                        output,
                                                        c_int(size), c_int(idx))
    if rtn != 0:
        return
    return output
#def dpuGetOutputTensorInCHWInt8(task, nodeName, data, size, idx=0):
#    """
#    Get DPU Task's output Tensor and store them under Caffe
#    Blob's order (channel/height/width) in INT8 format
#    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
#    nodeName: The pointer to DPU Node's name
#    data:     The output Tensor data
#    size:     The size (in Bytes) of output data to be stored
#    idx:      The index of a single output tensor for the Node, with default value of 0
#    Returns:  0 on success, or report error in case of failure
#    """
#    pyc_libn2cube.pyc_dpuGetOutputTensorInCHWInt8.argtypes = (
#        POINTER(c_void_p),c_char_p, np.ctypeslib.ndpointer(c_int8), c_int, c_int)
#    try:
#        rtn = pyc_libn2cube.pyc_dpuGetOutputTensorInCHWInt8(task, c_char_p(nodeName.encode("utf-8")),data,
#                                          c_int(size),
#                                          c_int(idx))
#    except Exception:
#        print('Please input data as dtype=np.int8')
#        return -1
#    return rtn

def dpuGetOutputTensorInCHWFP32(task, nodeName,  size, idx=0):
    """
    Get DPU Task's output Tensor and store them
    Blob's order (channel/height/width) in FP32 format
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    size:     The size (in Bytes) of output data to be stored
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The output Tensor's data
    """
    output = np.zeros(size, dtype = np.float32)
    rtn = pyc_libn2cube.pyc_dpuGetOutputTensorInCHWFP32(task,
                                                        c_char_p(nodeName.encode("utf-8")),
                                                        output,
                                                        c_int(size), c_int(idx))
    if rtn != 0:
        return
    return output

#def dpuGetOutputTensorInCHWFP32(task, nodeName, data, size, idx=0):
#    """
#    Get DPU Task's output Tensor and store them
#    Blob's order (channel/height/width) in FP32 format
#    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
#    nodeName: The pointer to DPU Node's name
#    data:     The output Tensor's data
#    size:     The size (in Bytes) of output data to be stored
#    idx:      The index of a single output tensor for the Node, with default value of 0
#    Returns:  0 on success, or report error in case of failure
#    """
#    pyc_libn2cube.pyc_dpuGetOutputTensorInCHWFP32.argtypes = (
#        POINTER(c_void_p),c_char_p, np.ctypeslib.ndpointer(c_float), c_int, c_int)
#    try:
#        rtn = pyc_libn2cube.pyc_dpuGetOutputTensorInCHWFP32(task, c_char_p(nodeName.encode("utf-8")),data,
#                                          c_int(size),
#                                          c_int(idx))
#    except Exception:
#        print('Please input data as dtype=np.float32')
#        return -1
#    return rtn


def dpuGetOutputTensorInHWCInt8(task, nodeName, size, idx=0):
    """
    Get DPU Task's output Tensor and store them under DPU
    Tensor's order (height/width/channel) in INT8 format
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    size:     The size (in Bytes) of output data to be stored
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The output Tensor's data
    """
    output = np.zeros(size, dtype = np.int8)
    rtn = pyc_libn2cube.pyc_dpuGetOutputTensorInHWCInt8(task,
                                                        c_char_p(nodeName.encode("utf-8")),
                                                        output,
                                                        c_int(size), c_int(idx))
    if rtn != 0:
        return
    return output
#def dpuGetOutputTensorInHWCInt8(task, nodeName, data, size, idx=0):
#    """
#    Get DPU Task's output Tensor and store them under DPU
#    Tensor's order (height/width/channel) in INT8 format
#    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
#    nodeName: The pointer to DPU Node's name
#    data:     The output Tensor's data
#    size:     The size (in Bytes) of output data to be stored
#    idx:      The index of a single output tensor for the Node, with default value of 0
#    Returns:  0 on success, or report error in case of failure
#    """
#    pyc_libn2cube.pyc_dpuGetOutputTensorInHWCInt8.argtypes = (
#        POINTER(c_void_p),c_char_p, np.ctypeslib.ndpointer(c_int8), c_int, c_int)
#    try:
#        rtn = pyc_libn2cube.pyc_dpuGetOutputTensorInHWCInt8(task, c_char_p(nodeName.encode("utf-8")),data,
#                                          c_int(size),
#                                          c_int(idx))
#    except Exception:
#        print('Please input data as dtype=np.int8')
#        return -1
#    return rtn

def dpuGetOutputTensorInHWCFP32(task, nodeName,  size, idx=0):
    """
    Get DPU Task's output Tensor and store them under DPU
    Tensor's order (height/width/channel) in FP32 format
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    size:     The size (in Bytes) of output data to be stored
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The output Tensor's data
    """
    output = np.zeros(size, dtype = np.float32)
    rtn = pyc_libn2cube.pyc_dpuGetOutputTensorInHWCFP32(task,
                                                        c_char_p(nodeName.encode("utf-8")),
                                                        output,
                                                        c_int(size), c_int(idx))
    if rtn != 0:
        return
    return output

#def dpuGetOutputTensorInHWCFP32(task, nodeName, data, size, idx=0):
#    """
#    Get DPU Task's output Tensor and store them under DPU
#    Tensor's order (height/width/channel) in FP32 format
#    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
#    nodeName: The pointer to DPU Node's name
#    data:     The output Tensor's data
#    size:     The size (in Bytes) of output data to be stored
#    idx:      The index of a single output tensor for the Node, with default value of 0
#    Returns:  0 on success, or report error in case of failure
#    """
#    pyc_libn2cube.pyc_dpuGetOutputTensorInHWCFP32.argtypes = (
#        POINTER(c_void_p),c_char_p, np.ctypeslib.ndpointer(c_float), c_int, c_int)
#    try:
#        rtn = pyc_libn2cube.pyc_dpuGetOutputTensorInHWCFP32(task, c_char_p(nodeName.encode("utf-8")),data,
#                                          c_int(size),
#                                          c_int(idx))
#    except Exception:
#        print('Please input data as dtype=np.float32')
#        return -1
#    return rtn

def dpuRunSoftmax(inputData, numClasses, batchSize, scale):
    """
    Compute softmax
    inputData:  Softmax input.
                This parameter should be gotten from the result of  dpuGetOuputTensorAddress()
    numClasses: The number of classes that softmax calculation operates on
    batchSize:  Batch size for the softmax calculation.
                This parameter should be specified with the division of the element number by inputs by numClasses
    scale:      The scale value applied to the input elements before softmax calculation
                This parameter typically can be obtained by using DNNDK API dpuGetRensorScale()
    Returns:    Result of softmax(numpy array)
    """
    output = np.zeros([batchSize*numClasses], dtype = np.float32)
    rtn = pyc_libn2cube.pyc_dpuRunSoftmax(inputData, output,
                                          c_int(numClasses),
                                          c_int(batchSize), c_float(scale))
    if rtn != 0:
        return
    return output

#def dpuRunSoftmax(inputData, outputData, numClasses, batchSize, scale):
#    """
#    Compute softmax
#    inputData:  Softmax input.
#                This parameter should be gotten from the result of  dpuGetOuputTensorAddress()
#    outputData: Result of softmax
#    numClasses: The number of classes that softmax calculation operates on
#    batchSize:  Batch size for the softmax calculation.
#                This parameter should be specified with the division of the element number by inputs by numClasses
#    scale:      The scale value applied to the input elements before softmax calculation
#                This parameter typically can be obtained by using DNNDK API dpuGetRensorScale()
#    Returns:    0 on success, or report error in case of failure
#    """
#
#    pyc_libn2cube.pyc_dpuRunSoftmax.argtypes = (
#        POINTER(c_int8), np.ctypeslib.ndpointer(c_float), c_int, c_int, c_float)
#    try:
#        rtn = pyc_libn2cube.pyc_dpuRunSoftmax(inputData, outputData,
#                                          c_int(numClasses),
#                                          c_int(batchSize), c_float(scale))
#    except Exception:
#        print('Please set outputData as dtype=np.float32')
#        return -1
#    return rtn


def dpuSetExceptionMode(mode):
    """
    Set the exception handling mode for DNNDK runtime N2Cube.
    It will affect all the APIs included in the libn2cube library
    mode:    The exception handling mode for runtime N2Cube to be specified.
             Available values include:
             -   N2CUBE_EXCEPTION_MODE_PRINT_AND_EXIT
             -   N2CUBE_EXCEPTION_MODE_RET_ERR_CODE
    Returns: 0 on success, or negative value in case of failure
    """
    return pyc_libn2cube.pyc_dpuSetExceptionMode(c_int(mode))


def dpuGetExceptionMode():
    """
    Get the exception handling mode for runtime N2Cube
    Returns: Current exception handing mode for N2Cube APIs.
             Available values include:
             -   N2CUBE_EXCEPTION_MODE_PRINT_AND_EXIT
             -   N2CUBE_EXCEPTION_MODE_RET_ERR_CODE
    """
    return pyc_libn2cube.pyc_dpuGetExceptionMode()


def dpuGetExceptionMessage(error_code):
    """
    Get the error message from error code (always negative value) returned by N2Cube APIs
    Returns: A pointer to a const string, indicating the error message for error_code
    """
    return pyc_libn2cube.dpuGetExceptionMessage(c_int(error_code))

def dpuSetTaskPriority(task, priority):
    """
    Set the priority of one DPU task. Priority range is 0 to 15, 0 has the highest priority.
    The priority of the task when it was created defaults to 15.
    """
    return pyc_libn2cube.pyc_dpuSetTaskPriority(task, c_int(priority))

def dpuGetTaskPriority(task):
    """
    Get the priority of one DPU task. Priority range is 0 to 15, 0 has the highest priority.
    The priority of the task when it was created defaults to 15.
    """
    return pyc_libn2cube.pyc_dpuGetTaskPriority(task)

def dpuSetTaskAffinity(task, coreMask):
    """
    Set the core mask of the task binding. Each bit represents a core.
    The default value of coreMask is 0x7 when the number of cores is 3.
    """
    return pyc_libn2cube.pyc_dpuSetTaskAffinity(task, c_int(coreMask))

def dpuGetTaskAffinity(task):
    """
    Get the core mask of the task binding. Each bit represents a core.
    The default value of coreMask is 0x7 when the number of cores is 3.
    """
    return pyc_libn2cube.pyc_dpuGetTaskAffinity(task)

def dpuGetInputTotalSize(task):
    """
    Get the total size in memory of all the inputs of the network,
    which involve the interleaves between two inputs.
    """
    return pyc_libn2cube.pyc_dpuGetInputTotalSize(task)

def dpuGetOutputTotalSize(task):
    """
    Get the total size in memory of all the outputs of the network
    which involve the interleaves between two outputs
    """
    return pyc_libn2cube.pyc_dpuGetOutputTotalSize(task)

def dpuGetBoundaryIOTensor(task, tensorName):
    return pyc_libn2cube.pyc_dpuGetBoundaryIOTensor(task,c_char_p(tensorName.encode("utf-8")))

