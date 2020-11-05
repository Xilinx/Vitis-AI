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
import cv2
import numpy as np
from dnndk import n2cube

try:
    pyc_libdputils = cdll.LoadLibrary("libn2cube.so")
except Exception:
    print('Load libn2cube.so failed\nPlease install DNNDK first!')


def dpuSetInputImageWithScale(task, nodeName, image, mean, scale, idx=0):
    """Set image into DPU Task's input Tensor with a specified scale parameter"""
    height = n2cube.dpuGetInputTensorHeight(task, nodeName, idx)
    width = n2cube.dpuGetInputTensorWidth(task, nodeName, idx)
    channel = n2cube.dpuGetInputTensorChannel(task, nodeName, idx)
    (imageHeight, imageWidth, imageChannel) = image.shape
    inputMean = (c_float * channel)()
    for i in range(0, channel):
        inputMean[i] = mean[i]

    if height == imageHeight and width == imageWidth:
        newImage = image
    else:
        newImage = cv2.resize(image, (width, height), 0, 0, cv2.INTER_LINEAR)

    inputImage = np.asarray(newImage, dtype=np.byte)
    inputImage2 = inputImage.ctypes.data_as(c_char_p)
    return pyc_libdputils.pyc_dpuSetInputData(task,
                                              c_char_p(nodeName.encode("utf-8")), inputImage2,
                                              c_int(height),
                                              c_int(width),
                                              c_int(imageChannel), inputMean,
                                              c_float(scale), c_int(idx))


def dpuSetInputImage(task, nodeName, image, mean, idx=0):
    """
    Set image into DPU Task's input Tensor
    task: DPU Task
    nodeName: The pointer to DPU Node name.
    image:    Input image in OpenCV Mat format. Single channel and 3-channel input image are supported.
    mean:     Mean value array which contains 1 member for single channel input image
              or 3 members for 3-channel input image
              Note: You can get the mean values from the input Caffe prototxt.
                    At present, the format of mean value file is not yet supported
    idx:      The index of a single input tensor for the Node, with default value as 0
    """
    return dpuSetInputImageWithScale(task, nodeName, image, mean, 1.0, idx)


def dpuSetInputImage2(task, nodeName, image, idx=0):
    """
    Set image into DPU Task's input Tensor (mean values automatically processed by N2Cube)
    nodeName: The pointer to DPU Node name.
    image:    Input image in OpenCV Mat format. Single channel and 3-channel input image are supported.
    idx:      The index of a single input tensor for the Node, with default value as 0
    """
    channel = n2cube.dpuGetInputTensorChannel(task, nodeName, idx)
    output = (c_float * channel)()
    outputMean = POINTER(c_float)(output)
    pyc_libdputils.loadMean(task, outputMean, channel)
    for i in range(channel):
        outputMean[i] = float(outputMean[i])
    return dpuSetInputImageWithScale(task, nodeName, image, outputMean, 1.0,
                                     idx)
