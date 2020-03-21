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

import graph_input_fn
from dnndk import n2cube
import numpy as np
from numpy import float32
import os

"""DPU Kernel Name for miniResNet"""
KERNEL_CONV="miniResNet_0"

CONV_INPUT_NODE="batch_normalization_1_FusedBatchNorm_1_add"
CONV_OUTPUT_NODE="dense_1_MatMul"

def get_script_directory():
    path = os.getcwd()
    return path

SCRIPT_DIR = get_script_directory()
calib_image_dir  = SCRIPT_DIR + "/../dataset/image_32_32/"
calib_image_list = calib_image_dir +  "words.txt"

def TopK(dataInput, filePath):
    """
    Get top k results according to its probability
    """
    cnt = [i for i in range(10)]
    pair = zip(dataInput, cnt)
    pair = sorted(pair, reverse=True)
    softmax_new, cnt_new = zip(*pair)
    #print(softmax_new,'\n',cnt_new)
    fp = open(filePath, "r")
    data1 = fp.readlines()
    fp.close()
    for i in range(5):
        flag = 0
        for line in data1:
            if flag == cnt_new[i]:
                print("Top[%d] %f %s" %(i, (softmax_new[i]),(line.strip)("\n")))
            flag = flag + 1

def main():

    """ Attach to DPU driver and prepare for running """
    n2cube.dpuOpen()

    """ Create DPU Kernels for CONV NODE in imniResNet """
    kernel = n2cube.dpuLoadKernel(KERNEL_CONV)

    """ Create DPU Tasks for CONV NODE in miniResNet """
    task = n2cube.dpuCreateTask(kernel, 0)

    listimage = os.listdir(calib_image_dir)

    for i in range(len(listimage)):
        path = os.path.join(calib_image_dir, listimage[i])
        if os.path.splitext(path)[1] != ".png":
            continue
        print("Loading %s" %listimage[i])

        """ Load image and Set image into CONV Task """
        imageRun=graph_input_fn.calib_input(path)
        imageRun=imageRun.reshape((imageRun.shape[0]*imageRun.shape[1]*imageRun.shape[2]))
        input_len=len(imageRun)
        n2cube.dpuSetInputTensorInHWCFP32(task,CONV_INPUT_NODE,imageRun,input_len)

        """  Launch miniRetNet task """
        n2cube.dpuRunTask(task)

        """ Get output tensor address of CONV """
        conf = n2cube.dpuGetOutputTensorAddress(task, CONV_OUTPUT_NODE)
        
        """ Get output channel of CONV  """
        channel = n2cube.dpuGetOutputTensorChannel(task, CONV_OUTPUT_NODE)
        
        """ Get output size of CONV  """
        size = n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE)
        
        softmax = np.zeros(size,dtype=np.float32)
       
        """ Get output scale of CONV  """
        scale = n2cube.dpuGetOutputTensorScale(task, CONV_OUTPUT_NODE)
        
        batchSize=size//channel
        """ Calculate softmax and show TOP5 classification result """
        softmax = n2cube.dpuRunSoftmax(conf, channel, batchSize, scale)
        TopK(softmax, calib_image_list)

    """ Destroy DPU Tasks & free resources """
    n2cube.dpuDestroyTask(task)
    """ Destroy DPU Kernels & free resources """
    rtn = n2cube.dpuDestroyKernel(kernel)
    """ Dettach from DPU driver & free resources """
    n2cube.dpuClose()
if __name__ == "__main__":
    main()
