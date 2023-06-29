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
from typing import List
import cv2
import numpy as np
import sys
import math
import xir
import vitis_ai_library
"""
 pre-process for resnet_v1_50_tf (tensorflow)
"""
_B_MEAN = 103.94
_G_MEAN = 116.78
_R_MEAN = 123.68
MEANS = [_B_MEAN, _G_MEAN, _R_MEAN]
SCALES = [1.0, 1.0, 1.0]

def resize_shortest_edge(image, smallest_side):
    H, W = image.shape[:2]
    if H >= W:
        nW = smallest_side
        nH = int(float(H)/W * smallest_side)
    else:
        nH = smallest_side
        nW = int(float(W)/H * smallest_side)
    #print("nW:nH=",nW,nH)
    return cv2.resize(image,(nW,nH))

def crop_image(image, height, width):
    image_height = image.shape[0]
    image_width = image.shape[1]
    offset_height = (image_height - height) // 2
    offset_width = (image_width - width) // 2
    return image[offset_height:offset_height + height, offset_width:offset_width + width, :]

def preprocess_one_image(image_path, width, height, means, scales, fixpos):
    image = cv2.imread(image_path)
    image = resize_shortest_edge(image, 256)
    image = crop_image(image, height, width)
    B, G, R = cv2.split(image)
    fix_scale = 2**fixpos
    B = (B - means[0]) * (scales[0] * fix_scale)
    G = (G - means[1]) * (scales[1] * fix_scale)
    R = (R - means[2]) * (scales[2] * fix_scale)
    image = cv2.merge([R, G, B])
    image = image.astype(np.int8)
    return image


"""
Get topk results according to its probability
datain: data result of graph_runner
filePath: filePath in which that records the infomation of kinds
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
        idx = 0
        for line in data1:
            if idx == cnt_new[i]:
                print("Top[%d] %d %f %s" % (i, idx, softmax_new[i], (line.strip)("\n")))
            idx = idx + 1


def main(argv):
    """create_graph_runner """
    g = xir.Graph.deserialize(argv[1])
    runner = vitis_ai_library.GraphRunner.create_graph_runner(g)
    """get_inputs & get_outputs"""
    input_tensor_buffers = runner.get_inputs()
    output_tensor_buffers = runner.get_outputs()

    input_ndim = tuple(input_tensor_buffers[0].get_tensor().dims)
    batch = input_ndim[0]
    height = input_ndim[1]
    width = input_ndim[2]
    fixpos = input_tensor_buffers[0].get_tensor().get_attr("fix_point")
    """init input data """
    image = preprocess_one_image(argv[2], width, height, MEANS, SCALES, fixpos)
    input_data = np.asarray(input_tensor_buffers[0])
    input_data[0] = image
    """ run graph runner"""
    job_id = runner.execute_async(input_tensor_buffers, output_tensor_buffers)
    runner.wait(job_id)

    pre_output_size = int(
        output_tensor_buffers[0].get_tensor().get_element_num() / batch)
    output_data = np.asarray(output_tensor_buffers[0])
    """post-process"""
    TopK(output_data[0], pre_output_size, "./words.txt")
    del runner


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("please model file and input file.")
    else:
        main(sys.argv)
