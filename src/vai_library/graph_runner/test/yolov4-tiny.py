#
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
#!/usr/bin/python3

import hashlib
import xir
import vart
import numpy as np


def md5(np_array):
    hash_md5 = hashlib.md5()
    hash_md5.update(np_array)
    return hash_md5.hexdigest()


g = xir.Graph.deserialize(
    '/workspace/aisw/debug_models/tensorflow-yolov4-tiny-master/tensorflow-yolov4-tiny-master/compile/yolov4-tiny/yolov4-tiny.xmodel'
)

# dissection of subgraphs.
subgraphs = g.get_root_subgraph().toposort_child_subgraph()
dpu_subgraph1 = subgraphs[2]
print("dpu_subgraph1 = " + dpu_subgraph1.get_name()
      )  # must be subgraph_detector/yolo-v4-tiny/Conv/Conv2D
dpu_subgraph2 = subgraphs[4]
print("dpu_subgraph2 = " + dpu_subgraph2.get_name()
      )  # must be subgraph_detector/yolo-v4-tiny/Conv_3/Conv2D

dpu_subgraph3 = subgraphs[6]
print("dpu_subgraph3 = " + dpu_subgraph3.get_name()
      )  # must be subgraph_detector/yolo-v4-tiny/Conv_10/Conv2D

dpu_subgraph4 = subgraphs[8]
print("dpu_subgraph4 = " + dpu_subgraph4.get_name()
      )  # must be subgraph_detector/yolo-v4-tiny/Conv_11/Conv2D

### start to run first DPU subgraph 'subgraph_detector/yolo-v4-tiny/Conv/Conv2D'
input1 = np.fromfile(
    '/scratch/models/cache/golden/74/32192dbe8b0cacdf99c2112732324b',
    dtype='int8')
print("md5(input1)={}".format(md5(input1)))  # 7432192dbe8b0cacdf99c2112732324b
input1 = input1.reshape([1, 416, 416, 3])
output1 = np.zeros(
    [1, 104, 104, 64], dtype='int8'
)  # it would be better to use fix point, convenient for comparing.
dpu_1 = vart.Runner.create_runner(dpu_subgraph1, "run")
job1 = dpu_1.execute_async([input1], [output1])
dpu_1.wait(job1)
print("md5(output1)={}".format(
    md5(output1)))  # a47ffd19dbae3b7185f48198e024736a

### start to run second DPU subgraph subgraph_detector/yolo-v4-tiny/Conv_3/Conv2D
### note this subgraph needs two inputs.
# copy is important, otherwise we see error like 'ndarray is not C-contiguous'
input2_0 = output1[:, :, :, 32:64].copy()
print("md5(input2_0)={}".format(
    md5(input2_0)))  # aa55fc2bfef038563e5a031dbddebee9
input2_1 = output1  # dpu2 need two inputs
output2 = np.zeros(
    [1, 52, 52, 128], dtype='int8'
)  # it would be better to use fix point, convenient for comparing.
dpu_2 = vart.Runner.create_runner(dpu_subgraph2, "run")
job2 = dpu_2.execute_async([input2_0, input2_1], [output2])
dpu_2.wait(job2)
print("md5(output2)={}".format(
    md5(output2)))  # 1866755506ebdb54c7f766fd530e1cc3

### start to run 3rd DPU subgraph subgraph_detector/yolo-v4-tiny/Conv_10/Conv2D
### similiar to the second subgraph.
input3_0 = output2[:, :, :, 64:128].copy()
print("md5(input3_0)={}".format(
    md5(input3_0)))  # 9fe461a5deb61f09210bb4ac415ec8b7
input3_1 = output2  # dpu3 need two inputs
output3 = np.zeros(
    [1, 26, 26, 256], dtype='int8'
)  # it would be better to use fix point, convenient for comparing.
dpu_3 = vart.Runner.create_runner(dpu_subgraph3, "run")
print("dpu_3.get_input_tensors()={}".format(dpu_3.get_input_tensors()))
# note: the input tensors do not have stable order, we must be careful to match the order of inputs.
job3 = dpu_3.execute_async([input3_1, input3_0], [output3])
dpu_3.wait(job3)
print("md5(output3)={}".format(
    md5(output3)))  # 4efe5a9bf47ce2bd861632ec1a535b34

### start to run 4th DPU subgraph subgraph_detector/yolo-v4-tiny/Conv_11/Conv2D
input4_0 = output3[:, :, :, 128:256].copy()
print("md5(input4_0)={}".format(
    md5(input4_0)))  # b4eb64306980a99f951ae2396edc08e4
input4_1 = output3  # dpu3 need two inputs
output4 = np.zeros(
    [1, 26, 26, 255], dtype='int8'
)  # it would be better to use fix point, convenient for comparing.
dpu_4 = vart.Runner.create_runner(dpu_subgraph4, "run")
print("dpu_4.get_input_tensors()={}".format(dpu_4.get_input_tensors()))
# note: the input tensors do not have stable order, we must be careful to match the order of inputs.
job4 = dpu_4.execute_async([input4_1, input4_0], [output4])
dpu_4.wait(job4)
print("md5(output4)={}".format(
    md5(output4)))  # 17eb158cbb6c978bb75445c3002998fb
