"""
Copyright 2022-2023 Advanced Micro Devices Inc.

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
import vart
import pathlib
import xir
import os
import math
import threading
import time
import sys

"""
Calculate softmax
data: data to be calculated
size: data size
return: softamx result
"""


def CPUCalcSoftmax(data, size, scale):
    sum = 0.0
    result = [0 for i in range(size)]
    for i in range(size):
        result[i] = math.exp(data[i] * scale)
        sum += result[i]
    for i in range(size):
        result[i] /= sum
    return result


def get_script_directory():
    path = os.getcwd()
    return path


"""
Get topk results according to its probability
datain: data result of softmax
filePath: filePath in witch that records the infotmation of kinds
"""

def TopK(datain, size, filePath):

    cnt = [i for i in range(size)]
    pair = zip(datain, cnt)
    pair = sorted(pair, reverse=True)
    softmax_new, cnt_new = zip(*pair)
    fp = open(filePath, "r")
    data1 = fp.readlines()
    fp.close()
    for i in range(5):
        flag = 0
        for line in data1:
            if (flag+1) == cnt_new[i]:
                print("Top[%d] %d %s" % (i, flag, (line.strip)("\n")))
            flag = flag + 1


"""
pre-process for resnet50 (caffe)
"""
_B_MEAN = 127.5
_G_MEAN = 127.5
_R_MEAN = 127.5
MEANS = [_B_MEAN, _G_MEAN, _R_MEAN]
SCALES = [0.007843137, 0.007843137, 0.007843137]

def resize_shortest_edge(image, size):
    H, W = image.shape[:2]
    if H >= W:
        nW = size
        nH = int(float(H)/W * size)
    else:
        nH = size
        nW = int(float(W)/H * size)
    #print("nW:nH=",nW,nH)
    return cv2.resize(image,(nW,nH))
def central_crop(image, crop_height, crop_width):
    image_height = image.shape[0]
    image_width = image.shape[1]
    offset_height = (image_height - crop_height) // 2
    offset_width = (image_width - crop_width) // 2
    return image[offset_height:offset_height + crop_height, offset_width:offset_width + crop_width, :]

def preprocess_one_image_fn(image_path, fix_scale, width=224, height=224):
    means = MEANS
    scales = SCALES
    image = cv2.imread(image_path)
    #image = cv2.resize(image, (width, height))
    image = resize_shortest_edge(image,256)
    image = central_crop(image, height, width)
    B, G, R = cv2.split(image)
    B = (B - means[0]) * scales[0] * fix_scale
    G = (G - means[1]) * scales[1] * fix_scale
    R = (R - means[2]) * scales[2] * fix_scale
    #image = cv2.merge([B, G, R])
    image = cv2.merge([R, G, B])
    image = image.astype(np.int8)
    return image


SCRIPT_DIR = get_script_directory()
calib_image_dir = SCRIPT_DIR + "/../images/"

global threadnum
threadnum = 0

"""
run inception_v1 with batch
dpu: dpu runner
img: imagelist to be run
cnt: threadnum
"""


def runInceptionV1(dpu: "Runner", img, cnt):
    """get tensor"""
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    shapeIn = tuple(inputTensors[0].dims)
    shapeOut = tuple(outputTensors[0].dims)
    pre_output_size = int(outputTensors[0].get_data_size() / shapeIn[0])

    output_fixpos = outputTensors[0].get_attr("fix_point")
    output_scale = 1 / (2**output_fixpos)
    count = 0
    n_of_images = len(img)
    while count < cnt:
        runSize = shapeIn[0]
        """prepare batch input/output """
        outputData = [np.empty(shapeOut, dtype=np.int8, order="C")]
        inputData = [np.empty(shapeIn, dtype=np.int8, order="C")]
        """init input image to input buffer """
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(shapeIn[1:])
        """run with batch """
        job_id = dpu.execute_async(inputData, outputData)
        dpu.wait(job_id)
        """softmax calculate with batch """
        """Benchmark DPU FPS performance over Vitis AI APIs execute_async() and wait() """
        """Uncomment the following code snippet to include softmax calculation for model’s end-to-end FPS evaluation """
        #for j in range(runSize):
        #    softmax = CPUCalcSoftmax(outputData[0][j], pre_output_size, output_scale)
        #    TopK(softmax, pre_output_size, "./words.txt")

        count = count + runSize


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

    """create runner """

    listimage = os.listdir(calib_image_dir)
    threadAll = []
    threadnum = int(argv[1])
    i = 0
    global runTotall
    runTotall = len(listimage)
    g = xir.Graph.deserialize(argv[2])
    subgraphs = get_child_subgraph_dpu(g)
    assert len(subgraphs) == 1  # only one DPU kernel

    all_dpu_runners = []
    for i in range(int(threadnum)):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))
    input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos
    """image list to be run """
    img = []
    for i in range(runTotall):
        path = os.path.join(calib_image_dir, listimage[i])
        img.append(preprocess_one_image_fn(path, input_scale))
    """
      The cnt variable is used to control the number of times a single-thread DPU runs.
      Users can modify the value according to actual needs. It is not recommended to use
      too small number when there are few input images, for example:
      1. If users can only provide very few images, e.g. only 1 image, they should set
         a relatively large number such as 360 to measure the average performance;
      2. If users provide a huge dataset, e.g. 50000 images in the directory, they can
         use the variable to control the test time, and no need to run the whole dataset.
    """
    cnt = 360
    """run with batch """
    time_start = time.time()
    for i in range(int(threadnum)):
        t1 = threading.Thread(
            target=runInceptionV1, args=(all_dpu_runners[i], img, cnt)
        )
        threadAll.append(t1)
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()

    del all_dpu_runners
    time_end = time.time()
    total = cnt * int(threadnum)
    timetotal = time_end - time_start
    fps = float(total / timetotal)
    print("%.2f FPS" % fps)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage : python3 inception_v1.py <thread_number> <inception_v1_model_file>")
    else:
        main(sys.argv)
