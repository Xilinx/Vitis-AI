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
import numpy as np

import demo.input
import demo.onnx
import demo.utils

parser = argparse.ArgumentParser(description="argparser")
parser.add_argument("--image_file_path", default="../orange_4.jpg")
parser.add_argument("--onnx_model_path", default="../squeezenet.onnx")
parser.add_argument("--class_file_path", default="./words.txt")

image_file_path = parser.parse_args().image_file_path
onnx_model_path = parser.parse_args().onnx_model_path
class_file_path = parser.parse_args().class_file_path

onnx_session = demo.onnx.OnnxSession(onnx_model_path)
model_shape = onnx_session.input_shape()
image_data = demo.input.InputData(image_file_path, model_shape).preprocess()
input_data = np.expand_dims(image_data, axis = 0)

raw_result = onnx_session.run(input_data)
res_list = demo.utils.softmax(raw_result)
sort_idx = demo.utils.sort_idx(res_list)

with open(class_file_path, "rt") as f:
    classes = f.read().rstrip('\n').split('\n')

print('============ Top 5 labels are: ============================')
for k in sort_idx[:5]:
    print(classes[k], res_list[k])
print('===========================================================')

