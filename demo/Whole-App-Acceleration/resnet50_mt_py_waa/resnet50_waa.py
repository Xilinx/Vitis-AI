"""
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
"""

from ctypes import *
from typing import List
import cv2
import numpy as np
import xir
import vart
import os
import math
import threading
import time
import sys
import waa_rt

"""
Calculate softmax
data: data to be calculated
size: data size
return: softamx result
"""


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


"""
Get topk results according to its probability
datain: data result of softmax
filePath: filePath in witch that records the infotmation of kinds
"""


def TopK(datain,size,filePath):

    cnt=[i for i in range(size) ]
    pair=zip(datain,cnt)
    pair=sorted(pair,reverse=True)
    softmax_new,cnt_new=zip(*pair)
    fp=open(filePath, "r")
    data1=fp.readlines()
    fp.close()
    for i in range(5):
        idx = 0
        for line in data1:
            if idx == cnt_new[i]:
                print("Top[%d] %d %s" % (i, idx, (line.strip)("\n")))
            idx = idx + 1
"""
pre-process for resnet50 (caffe)
"""
_B_MEAN = 104.0
_G_MEAN = 107.0
_R_MEAN = 123.0
MEANS = [_B_MEAN,_G_MEAN,_R_MEAN]
SCALES = [1.0, 1.0, 1.0]

def preprocess_one_image_fn(image_path, width=224, height=224):
   means = MEANS
   scales = SCALES
   image = cv2.imread(image_path)
   image = cv2.resize(image,(width, height))
   B, G, R = cv2.split(image)
   B = (B - means[0]) * scales[0]
   G = (G - means[1]) * scales[1]
   R = (R - means[2]) * scales[2]
   image = cv2.merge([B, G, R])
   return image


SCRIPT_DIR = get_script_directory()
calib_image_dir = SCRIPT_DIR + "/images/"
global threadnum
threadnum = 0

"""
run resnt50 with batch
runner: dpu runner
img: imagelist to be run
cnt: threadnum
"""


def runResnet50(runner: "Runner", img, cnt):
    """get tensor"""
    inputTensors = runner.get_input_tensors()
    outputTensors = runner.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    pre_output_size = int(outputTensors[0].get_data_size() / input_ndim[0])


    output_ndim = tuple(outputTensors[0].dims)
    n_of_images = len(img)
    count = 0
    while count < cnt:
        runSize = input_ndim[0]
        """prepare batch input/output """
        inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]
        outputData = [np.empty(output_ndim, dtype=np.float32, order="C")]

        """init input image to input buffer """
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])

        """run with batch """
        job_id = runner.execute_async(inputData, outputData)
        runner.wait(job_id)


        """softmax&TopK calculate with batch """
        """Benchmark DPU FPS performance over Vitis AI APIs execute_async() and wait() """
        """Uncomment the following code snippet to include softmax calculation for model’s end-to-end FPS evaluation """
        #for j in range(runSize):
        #    softmax = CPUCalcSoftmax(outputData[0][j], pre_output_size)
        #    TopK(softmax, pre_output_size, "./words.txt")
        count = count + runSize
"""
 obtain dpu subgrah
"""
def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (
        root_subgraph is not None
    ), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def main(argv):
    global threadnum

    listimage=os.listdir(calib_image_dir)
    threadAll = []
    threadnum = int(argv[1])
    i = 0
    global runTotall
    runTotall = len(listimage)
    g = xir.Graph.deserialize(argv[2])
    subgraphs = get_child_subgraph_dpu(g)
    assert len(subgraphs) == 1 # only one DPU kernel
    all_dpu_runners = []
    for i in range(int(threadnum)):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))
    """image list to be run """
    xclbin_p=str("/usr/lib/dpu.xclbin")
    kernelName_p="pp_pipeline_accel"
    deviceIdx_p=0
    fpga_pp = waa_rt.PreProcess(xclbin_p,kernelName_p,deviceIdx_p)
    time1 = int(round(time.time() * 1000))
    img = []
    time_start = time.time()
    for i in range(runTotall):
        path = os.path.join(calib_image_dir,listimage[i])
        img.append(fpga_pp.preprocess_input(path))

    cnt = 1
    """run with batch """
    for i in range(int(threadnum)):
        t1 = threading.Thread(target=runResnet50, args=(all_dpu_runners[i], img, cnt))
        threadAll.append(t1)
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()

    del all_dpu_runners
    #print("Pre time: %d ms" %(time_pre - time1))
    time_end = time.time()
    timetotal = time_end - time_start
    total_frames = runTotall
    fps = float(total_frames / timetotal)
    print(
        "FPS=%.2f, total frames = %.2f , time=%.6f seconds"
        % (fps, total_frames, timetotal)
    )
    #print("Total time : %d ms" %timetotal)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage : python3 resnet50.py <thread_number> <resnet50_xmodel_file>")
    else :
        main(sys.argv)
