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

import time
import multiprocessing
import os
from multiprocessing import Process
from multiprocessing import shared_memory
import argparse
import threading
from threading import Lock, Thread
import numpy as np

import demo.input
import demo.onnx
import demo.utils

parser = argparse.ArgumentParser(description="argparser")
parser.add_argument("--image_file_path", default="../orange_4.jpg")
parser.add_argument("--onnx_model_path", default="../squeezenet.onnx")
parser.add_argument("--class_file_path", default="./words.txt")
parser.add_argument("--duration", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=1)

image_file_path = parser.parse_args().image_file_path
onnx_model_path = parser.parse_args().onnx_model_path
class_file_path = parser.parse_args().class_file_path
duration = parser.parse_args().duration
default_batch_size = parser.parse_args().batch_size
proc_count = os.cpu_count()
proc_count = 1
exit = False

task_count = [0 for x in range(0, proc_count)]

def func(idx, task_count):
    onnx_session = demo.onnx.OnnxSession(onnx_model_path)
    model_shape = onnx_session.input_shape()
    batch_size = default_batch_size
    if isinstance(model_shape[0], int):
        batch_size = model_shape[0]
    image_data = demo.input.InputData(image_file_path, model_shape).preprocess()
    input_data = np.repeat(np.expand_dims(image_data, axis = 0), batch_size, 0)
    while not exit:
        raw_result = onnx_session.run(input_data)
        task_count[idx] += batch_size

thread_list = []
for i in range(proc_count):
    t = threading.Thread(target = func, args = (i, task_count))
    t.start()
    thread_list.append(t)

time.sleep(duration)

exit = True
N = 0

for i in range(proc_count):
    thread_list[i].join()
    N += task_count[i]

print("qps : ", N / duration, "/s")
