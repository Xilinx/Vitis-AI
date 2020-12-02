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

from ctypes import *
import cv2
import numpy as np
import runner
import os
import input_fn
import math
import threading
import time
import sys

'''
Calculate softmax
data: data to be calculated
size: data size
return: softamx result
'''
def CPUCalcSoftmax(data,size):
    sum=0.0
    result = [0 for i in range(size)]
    for i in range(size):
        result[i] = math.exp(data[i])
        sum +=result[i]
    for i in range(size):
        result[i] /=sum
    return result

def get_script_directory():
    path = os.getcwd()
    return path

'''
Get topk results according to its probability
datain: data result of softmax
filePath: filePath in witch that records the infotmation of kinds
'''
def TopK(datain, size, filePath):

    cnt  = [i for i in range(size) ]
    pair = zip(datain, cnt)
    pair = sorted(pair, reverse=True)
    softmax_new, cnt_new = zip(*pair)
    fp = open(filePath, "r")
    data1 = fp.readlines()
    fp.close()
    print ("")
    for i in range(5):
        flag = 0
        for line in data1:
            if flag == cnt_new[i]:
                print("Top[%d] %f %s" % (i, (softmax_new[i]), (line.strip)("\n")))
            flag = flag+1

l = threading.Lock()
SCRIPT_DIR = get_script_directory()
calib_image_dir  = "./image"
label_file = "./words.txt"
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
batchSize = 4
global threadnum
threadnum = 0
global runTotall
runRotal = 0

'''
run resnt50 with batch
dpu: dpu runner
img: imagelist to be run
cnt: threadnum
'''
def runResnet50(dpu, img, cnt):

    """get tensor"""
    inputTensors  = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    tensorformat  = dpu.get_tensor_format()
    if tensorformat == dpu.TensorFormat.NCHW:
        outputHeight  = outputTensors[0].dims[2]
        outputWidth   = outputTensors[0].dims[3]
        outputChannel = outputTensors[0].dims[1]
    elif tensorformat == dpu.TensorFormat.NHWC:
        outputHeight  = outputTensors[0].dims[1]
        outputWidth   = outputTensors[0].dims[2]
        outputChannel = outputTensors[0].dims[3]
    else:
        exit("Format error")
    outputSize = outputHeight * outputWidth * outputChannel
    softmax = np.empty(outputSize)

    global runTotall

    count = cnt

    while count < runTotall:
        l.acquire()
        if (runTotall < (count+batchSize)):
            runSize = runTotall - count
        else:
            runSize = batchSize
        l.release()
        shapeIn = (runSize,) + tuple([inputTensors[0].dims[i] for i in range(inputTensors[0].ndims)][1:])

        """prepare batch input/output """
        outputData = []
        inputData  = []
        outputData.append(np.empty((runSize,outputSize), dtype = np.float32, order = 'C'))
        inputData.append(np.empty((shapeIn), dtype = np.float32, order = 'C'))

        """init input image to input buffer """
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j,...] = img[count+j]

        """run with batch """
        job_id = dpu.execute_async(inputData,outputData)
        dpu.wait(job_id)

        """softmax calculate with batch """
        for j in range(runSize):
            softmax = CPUCalcSoftmax(outputData[0][j], outputSize)
            TopK(softmax, outputSize, label_file)
        l.acquire()
        count = count + threadnum*runSize
        l.release()

def main(argv):
    global threadnum

    """create runner """
    dpu = runner.Runner(argv[2])

    listimage = os.listdir(calib_image_dir)
    threadAll = []
    threadnum = int(argv[1])
    i = 0
    global runTotall
    runTotall = len(listimage)

    """image list to be run """
    img = []
    for i in range(runTotall):
        path = os.path.join(calib_image_dir, listimage[i])
        img.append(input_fn.preprocess_fn(path))

    imgData = np.transpose(img, (0, 3, 1, 2))
    """run with batch """
    time1 = time.time()
    for i in range(int(threadnum)):
        t1 = threading.Thread(target=runResnet50, args=(dpu, imgData, i*batchSize))
        threadAll.append(t1)
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()

    time2 = time.time()

    timetotal = time2 - time1
    fps = float(runTotall / timetotal)
    print("%.2f FPS" %fps)

    del dpu

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("please input thread number and json file path.")
    else :
        main(sys.argv)
