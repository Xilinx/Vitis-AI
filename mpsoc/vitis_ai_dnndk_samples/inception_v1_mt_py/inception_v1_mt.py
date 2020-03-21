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
from numpy import float32
from dnndk import n2cube
import os
import threading
import time
import sys

sys.path.append("..")
from common import dputils 

l = threading.Lock()

def RunDPU(kernel, img, count):
    """
    DPU run function
    kernel: dpu kernel
    img: image to be run
    count : test rounds count
    """
    """Create DPU Tasks from DPU Kernel"""
    task = n2cube.dpuCreateTask(kernel, 0)
    while count < 1000:
        """Load image to DPU"""
        dputils.dpuSetInputImage2(task, KERNEL_CONV_INPUT, img)
        
        """Get input Tesor"""
        tensor = n2cube.dpuGetInputTensor(task, KERNEL_CONV_INPUT)
        
        """Model run on DPU"""
        n2cube.dpuRunTask(task)
        
        """Get the output tensor size from FC output"""
        size = n2cube.dpuGetOutputTensorSize(task, KERNEL_FC_OUTPUT)
        
        """Get the output tensor channel from FC output"""
        channel = n2cube.dpuGetOutputTensorChannel(task, KERNEL_FC_OUTPUT)
        
        softmax = np.zeros(size,dtype=float32)
        
        """Get FC result"""
        conf = n2cube.dpuGetOutputTensorAddress(task, KERNEL_FC_OUTPUT)
        
        """Get output scale of FC"""
        outputScale = n2cube.dpuGetOutputTensorScale(task, KERNEL_FC_OUTPUT)
        
        """Run softmax"""
        softmax = n2cube.dpuRunSoftmax(conf, channel, size // channel, outputScale)
        
        l.acquire()
        count = count + threadnum
        l.release()

    """Destroy DPU Tasks & free resources"""
    n2cube.dpuDestroyTask(task)

global threadnum
threadnum = 0
count = 0
KERNEL_CONV = "inception_v1_0"
KERNEL_CONV_INPUT = "conv1_7x7_s2"
KERNEL_FC_OUTPUT = "loss3_classifier"

"""
brief Entry for runing GoogLeNet neural network
"""
def main(argv):

    """Attach to DPU driver and prepare for runing"""
    n2cube.dpuOpen()

    """Create DPU Kernels for GoogLeNet"""
    kernel = n2cube.dpuLoadKernel(KERNEL_CONV)

    image_path = "./../dataset/image_224_224/"
    
    listimage = os.listdir(image_path)
    
    path = os.path.join(image_path, listimage[0])
    
    print("Loading  %s" %listimage[0])
    
    img = cv2.imread(path)
    
    threadAll = []
    global threadnum
    threadnum = int(argv[1])
    print("Input thread number is: %d" %threadnum)
    
    time1 = time.time()
    
    for i in range(int(threadnum)):
        t1 = threading.Thread(target=RunDPU, args=(kernel, img, i))
        threadAll.append(t1)
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    
    time2 = time.time()
    
    timetotal = time2 - time1
    fps = float(1000 / timetotal)
    print("%.2f FPS" %fps)

    """Destroy DPU Tasks & free resources"""
    rtn = n2cube.dpuDestroyKernel(kernel)

    """Dettach from DPU driver & release resources"""
    n2cube.dpuClose()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("please input thread number.")
    else :
        main(sys.argv)
