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
import argparse
import math
import threading
import time
import sys

def get_script_directory():
    path = os.getcwd()
    return path

def pad(volume, paddings, val):
    o_shape = volume.shape
    ss = len(paddings)//2 -1
    n_shape = []
    for i in range(ss+1):
      tmp = o_shape[i]+paddings[(ss-i)*2]+paddings[(ss-i)*2+1] 
      n_shape.append(tmp)
    volume_new = np.empty(n_shape, dtype=np.float32)
    volume_new.fill(val)
    volume_new[..., 
               paddings[4]:(n_shape[2]-paddings[5]),
               paddings[2]:(n_shape[3]-paddings[3]),
               paddings[0]:(n_shape[4]-paddings[1]),
               ] = volume
    return volume_new

def pad_input(volume, roi_shape, strides, padding_mode, padding_val, dim=3):
    bounds = [(strides[i] - volume.shape[2:][i] % strides[i]) % strides[i] for i in range(dim)]
    bounds = [bounds[i] if (volume.shape[2:][i] + bounds[i]) >= roi_shape[i] else bounds[i] + strides[i]
              for i in range(dim)]
    paddings = [bounds[2] // 2, bounds[2] - bounds[2] // 2,
                bounds[1] // 2, bounds[1] - bounds[1] // 2,
                bounds[0] // 2, bounds[0] - bounds[0] // 2,
                0, 0,
                0, 0]
    return pad(volume, paddings, padding_val), paddings

def gauss(M, std, sym = True):
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, 'd')
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = np.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = np.exp(-n ** 2 / sig2)
    if not sym and not odd:
        w = w[:-1]
    return w

def gaussian_kernel(n, std):
    gaussian1D = gauss(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    gaussian3D = np.outer(gaussian2D, gaussian1D)
    gaussian3D = gaussian3D.reshape(n, n, n)
    gaussian3D = np.cbrt(gaussian3D)
    gaussian3D /= gaussian3D.max()
    return gaussian3D

"""
macro definition for unet3d
"""
MEAN = 101.0
SCALE = 76.9

roi_shape = [128,128,128]
overlap=0.5
padding_val = -2.2

SCRIPT_DIR = get_script_directory()
calib_image_dir = "/group/xbjlab/dphi_software/software/workspace/huizhang/accuracy_test_library/unet3d/preprocessed_data/"
out_dir = "./output/"
mode = ""

def runUnet3d(runner: "Runner", listimage, cnt, arg):
    """image list to be run """
    path = []
    runTotall = len(listimage)
    for i in range(runTotall):
        if 'x' in listimage[i]:
            path.append(os.path.join(arg.i_dir, listimage[i]))
    """get tensor"""
    inputTensors = runner.get_input_tensors()
    outputTensors = runner.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    print("input dims:", input_ndim)
    pre_output_size = int(outputTensors[0].get_data_size() / input_ndim[0])

    output_ndim = tuple(outputTensors[0].dims)
    print("output dims:", output_ndim)
    output_fixpos = outputTensors[0].get_attr("fix_point")
    output_scale = 1 / (2**output_fixpos)

    input_fixpos = inputTensors[0].get_attr("fix_point")
    input_scale = 2**input_fixpos
    count = 0
    origin = np.array([])
    while count < cnt:
        if arg.perf:
            if origin.size == 0:
                origin = np.load(path[0])
            else:
                pass
        else:
            origin = np.load(path[count])
        runSize = input_ndim[0]
        inputs = np.expand_dims(origin, axis = 0)
        image_shape = list(inputs.shape[2:])
        dim = len(image_shape)
        strides = [int(roi_shape[i] * (1 - overlap)) for i in range(dim)]
        bounds = [image_shape[i] % strides[i] for i in range(dim)]
        bounds = [bounds[i] if bounds[i] < strides[i] // 2 else 0 for i in range(dim)]
        inputs = inputs[...,
             bounds[0] // 2: image_shape[0] - (bounds[0] - bounds[0] // 2),
             bounds[1] // 2: image_shape[1] - (bounds[1] - bounds[1] // 2),
             bounds[2] // 2: image_shape[2] - (bounds[2] - bounds[2] // 2)]
        inputs, paddings = pad_input(inputs, roi_shape, strides, "constant", padding_val)
        padded_shape = inputs.shape[2:]
        size = [(inputs.shape[2:][i] - roi_shape[i]) // strides[i] + 1 for i in range(dim)]
        inputs = inputs * input_scale
        inputs = np.around(inputs)
        inputs = inputs.astype(np.int8)

        result = np.zeros(shape=(1, 3, *padded_shape), dtype=np.float32)
        norm_patch = gaussian_kernel(roi_shape[0], 0.125*roi_shape[0]).astype(result.dtype)
        norm_map = np.zeros(result.shape, result.dtype)

        """prepare batch input/output """
        inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]
        outputData = [np.empty(output_ndim, dtype=np.int8, order="C")]
        """init input image to input buffer """
        cycle = size[0]*size[1]*size[2]
        x = 0
        b = 0
        time1 = time.time()
        ind = []
        for i in range(0, strides[0] * size[0], strides[0]):
            for j in range(0, strides[1] * size[1], strides[1]):
                for k in range(0, strides[2] * size[2], strides[2]):
                  imageRun = inputData[0]
                  imageRun[b, ...] = inputs[
                                           ...,
                                           i:(roi_shape[0] + i),
                                           j:(roi_shape[1] + j),
                                           k:(roi_shape[2] + k)
                                           ].reshape(input_ndim[1:]).transpose(1,2,0,3)
                  ind.append((i, j, k))
                  x = x + 1
                  b = b + 1
                  if (b == runSize or x == cycle):
                    """run with batch """
                    job_id = runner.execute_async(inputData, outputData)
                    runner.wait(job_id)

                    for m in range(0, b):
                      i,j,k = ind[m]
                      result[
                           ...,
                           i:(roi_shape[0] + i),
                           j:(roi_shape[1] + j),
                           k:(roi_shape[2] + k)] += outputData[0][m, ...].transpose(3,2,0,1).astype(np.float32) * output_scale * norm_patch
                      norm_map[
                           ...,
                           i:(roi_shape[0] + i),
                           j:(roi_shape[1] + j),
                           k:(roi_shape[2] + k)] += norm_patch
                    ind.clear()
                    b = 0
        result /= norm_map
        time2 = time.time()
        time_dpu = time2 - time1
        #print("dpu time=%.6f seconds" % (time_dpu))
 
        """Benchmark DPU FPS performance over Vitis AI APIs execute_async() and wait() """
        """Uncomment the following code snippet to include softmax calculation for model’s end-to-end FPS evaluation """
        #print("result shape: ", result.shape)
        result = result[                                                  
            ...,                                                     
            paddings[4]: image_shape[0] + paddings[4],               
            paddings[2]: image_shape[1] + paddings[2],               
            paddings[0]: image_shape[2] + paddings[0]
            ]
        name = listimage[count].split(".")[0]
        name = name.split("_")[1]
        if arg.acc:
            outname = arg.o_dir + name + ".npy"
            print(outname)
            np.save(outname, result)

        result = np.amax(result, axis=1)
        if arg.save_img:
            slic = result.shape[1]
            for l in range(0, slic, slic//6):
                vmin = result[0][l].min()
                vmax = result[0][l].max()
                scale = 256 / (vmax-vmin)
                save_name = name + "_" + str(l) + ".jpg"
                print("save picture: ", save_name)
                cv2.imwrite(save_name, (result[0][l] - vmin)*scale)

        count = count + 1

"""
 obtain dpu subgrah
"""
def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph
            is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def main(arg):
    listimage = os.listdir(arg.i_dir)
    threadAll = []
    threadnum = arg.threads
    i = 0
    g = xir.Graph.deserialize(arg.xmodel)
    subgraphs = get_child_subgraph_dpu(g)
    assert len(subgraphs) == 1  # only one DPU kernel
    all_dpu_runners = []
    for i in range(threadnum):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    """
      The cnt variable is used to control the number of times a single-thread DPU runs.
      Users can modify the value according to actual needs. It is not recommended to use
      too small number when there are few input images, for example:
      1. If users can only provide very few images, e.g. only 1 image, they should set
         a relatively large number such as 360 to measure the average performance;
      2. If users provide a huge dataset, e.g. 50000 images in the directory, they can
         use the variable to control the test time, and no need to run the whole dataset.
    """
    if arg.perf:
        cnt = arg.cycle
    else:
        cnt = len(listimage)
        #cnt = 1
    print("total test images: ", cnt)
    """run with batch """
    time_start = time.time()
    for i in range(threadnum):
        t1 = threading.Thread(target=runUnet3d, args=(all_dpu_runners[i], listimage, cnt, arg))
        threadAll.append(t1)
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()

    del all_dpu_runners

    time_end = time.time()
    timetotal = time_end - time_start
    total_frames = cnt * threadnum
    fps = float(total_frames / timetotal)
    print(
        "FPS=%.2f, total frames = %.2f , time=%.6f seconds"
        % (fps, total_frames, timetotal)
    )


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='Unet-3d dpu version')
    PARSER.add_argument('--input', '-i', dest='i_dir', type=str, default=calib_image_dir)
    PARSER.add_argument('--output', '-o', dest='o_dir', type=str, default=out_dir)
    PARSER.add_argument('--acc', '-a', dest='acc', help='test accuracy', action='store_true', default=False)
    PARSER.add_argument('--xmodel', '-x', dest='xmodel', type=str, default="")
    PARSER.add_argument('--save_img', '-s', dest='save_img', action='store_true', default=False)

    PARSER.add_argument('--performance', '-p', dest='perf', action='store_true', default=False)
    PARSER.add_argument('--threads', '-t', dest='threads', type=int, default=1)
    PARSER.add_argument('--cycle', '-c', dest='cycle', type=int, default=10)

    flags = PARSER.parse_args()

    main(flags)
