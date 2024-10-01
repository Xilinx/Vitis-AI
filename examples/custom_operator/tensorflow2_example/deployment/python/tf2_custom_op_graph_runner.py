'''
Copyright 2022-2023 Advanced Micro Devices Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations un:der the License.
'''

import cv2
import numpy as np
import os
import xir
import argparse
import vitis_ai_library

supported_ext = [".jpg", ".jpeg", ".png"]


def get_imagefiles(image_path, batchsize):
    files = [f for f in os.listdir(image_path) if f[f.rfind('.'):] in supported_ext]

    img_num = len(files)
    if (img_num % batchsize > 0):
        app_num = batchsize - img_num % batchsize
        while (app_num > 0):
            files.append(files[img_num - 1])
            app_num = app_num - 1

    if not image_path.endswith('/'):
        image_path = image_path + '/'

    return [image_path + f for f in files]


def preprocess_tf2_custom_op(imgfiles, begidx, input_tensor_buffers):
    input_tensor = input_tensor_buffers[0].get_tensor()
    batchsize = input_tensor.dims[0]
    height = input_tensor.dims[1]
    width = input_tensor.dims[2]

    fix_pos = input_tensor_buffers[0].get_tensor().get_attr("fix_point")
    scale = 2**fix_pos

    for i in range(batchsize):
        img = cv2.imread(imgfiles[begidx + i], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (width, height))

        img = np.array(img).reshape((width, height, 1))
        img = (img * scale).astype(np.int8)

        input_data = np.asarray(input_tensor_buffers[0])
        input_data[i][:] = img


def postprocess_tf2_custom_op(imgfiles, begidx, output_tensor_buffers):
    output_data = np.array(output_tensor_buffers[0])

    batchsize = output_tensor_buffers[0].get_tensor().dims[0]
    element = output_tensor_buffers[0].get_tensor().get_element_num() // batchsize

    for i in range(batchsize):
        print("image file", imgfiles[begidx + i], "result:")
        for e in range(element):
            print("\tscore[", e, "] = ", output_data[i][e])


def app(image_path, model):
    g = xir.Graph.deserialize(model)
    runner = vitis_ai_library.GraphRunner.create_graph_runner(g)
    print("graph runner created.")

    input_tensor_buffers = runner.get_inputs()
    output_tensor_buffers = runner.get_outputs()

    batchsize = input_tensor_buffers[0].get_tensor().dims[0]
    files = get_imagefiles(image_path, batchsize)

    loop_num = len(files) // batchsize
    for i in range(loop_num):
        begin_idx = i * batchsize
        preprocess_tf2_custom_op(files, begin_idx, input_tensor_buffers)

        v = runner.execute_async(input_tensor_buffers, output_tensor_buffers)
        runner.wait(v)

        postprocess_tf2_custom_op(files, begin_idx, output_tensor_buffers)

    del runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', type=str, help='xmodel name')
    ap.add_argument('image_path', type=str, help='image path')

    args = ap.parse_args()

    print('Command line options:')
    print(' model      : ', args.model)
    print(' image_path : ', args.image_path)

    app(args.image_path, args.model)


if __name__ == '__main__':
    main()
