# Copyright 2022-2023 Advanced Micro Devices Inc.
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

import argparse
import os

import numpy as np

import demo.input
import demo.onnx
import demo.utils

parser = argparse.ArgumentParser(description="argparser")
parser.add_argument("--onnx_model_path", default="../squeezenet.onnx")
parser.add_argument("--dataset_path", default="dataset/")
parser.add_argument("--val_path", default="val.txt")


image_labels = {}
onnx_model_path = parser.parse_args().onnx_model_path
with open(parser.parse_args().val_path) as val_file:
    for line in val_file:
        val = line.strip(" ").split()
        image_file_path = val[0]
        label = int(val[1])
        image_labels[image_file_path] = label

onnx_session = demo.onnx.OnnxSession(onnx_model_path)
model_shape = onnx_session.input_shape()

image_count = 0
top1_correct = 0
top5_correct = 0
for root, dirs, files in os.walk(parser.parse_args().dataset_path):
    for f in files:
        image_count = image_count + 1

        image_file_path = os.path.join(root, f)
        image_data = demo.input.InputData(image_file_path, model_shape).preprocess()
        input_data = np.expand_dims(image_data, axis = 0)
        raw_result = onnx_session.run(input_data)
        res_list = demo.utils.softmax(raw_result)
        sort_idx = demo.utils.sort_idx(res_list)
        target_label = image_labels[f]

        if (target_label == sort_idx[0]):
            top1_correct = top1_correct + 1
        if (target_label in sort_idx[:5]):
            top5_correct = top5_correct + 1

print("top1 accuracy : ", top1_correct / image_count)
print("top5 accuracy : ", top5_correct / image_count)
