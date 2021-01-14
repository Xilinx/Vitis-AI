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

import dpu4rnn_py
import numpy as np

nums = []
file_i = open("./fix_input.txt", 'r')
file_o = open("./the_output.txt", 'w')
lines = file_i.readlines()
for line in lines:
    num_str = line.split(' ')
    num_new = list()
    for n in num_str:
        num_new.append(int(n))
    nums.append(num_new)
file_i.close()

frame_num = 59 #36
a = dpu4rnn_py.dpu4rnn.create('openie')
input_num = np.array(nums)
input_num = input_num.astype(np.int16)
zeros_cat = np.zeros((frame_num, 24), dtype=np.int16)
input_num = np.concatenate((input_num, zeros_cat), axis = 1)
output_num = np.zeros(300*frame_num, dtype=np.int16)
print(input_num.shape)
print(output_num.shape)
a.run(input_num.flatten(), 224*frame_num*2, output_num, frame_num)
#output_num = output_num.reshape(frame_num, 300)
print(output_num)
x = 0
for i in range(0, frame_num):
    for j in range(0, 300):
        file_o.write(str(output_num[x]))
        file_o.write(' ')
        x=x+1
    file_o.write('\n')
file_o.close()

b = a.getBatch()
print("xxxxx: ", b)
