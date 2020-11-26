

#
# Copyright 2019 Xilinx Inc.
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
#

import os
import numpy as np
import math
from ctypes import *

def read_is(layer_name_list, line_number_3, line_number_4, addr, SAVE_PATH):

  print("=========begin to generate instr bin==========\n")
  
  batch3_in     = os.path.join(SAVE_PATH, "ddr_bin_3k")
  batch3_is_out = os.path.join(SAVE_PATH, "is_3")
  batch4_in     = os.path.join(SAVE_PATH, "ddr_bin_4k")
  batch4_is_out = os.path.join(SAVE_PATH, "is_4")

  layer_num = len(layer_name_list)

  line_dim3 = len(line_number_3)
  line_3 = (c_uint*line_dim3)()
  for i in range(0, len(line_number_3)):
    line_3[i] = line_number_3[i]

  line_dim4 = len(line_number_4)
  line_4 = (c_uint*line_dim4)()
  for i in range(0, len(line_number_4)):
    line_4[i] = line_number_4[i]

  if layer_num == 1:  
    is_3_size = ( (layer_num*(line_3[0]+line_3[1]+(3*3)+2)) + line_3[line_dim3-1] + 1)*0x10 # +1 :add 1 line(4*32bits) of byass regs 
    print("Size of instr batch3: {} ".format(is_3_size))  
    is_4_size = ( (layer_num*(line_4[0]+line_4[1]+(3*4)+2)) + line_4[line_dim4-1] + 1)*0x10
    print("Size of instr batch4: {} ".format(is_4_size))
  else:
    is_3_size = ( (layer_num*(line_3[0]+line_3[1]+(3*3)+2)) + 1 + 1)*0x10 # +1 +1 :add 1 line(4*32bits) of byass regs, add 1 line of end instr
    print("Size of instr batch3: {} ".format(is_3_size))
    is_4_size = ( (layer_num*(line_4[0]+line_4[1]+(3*4)+2)) + 1 + 1)*0x10
    print("Size of instr batch4: {} ".format(is_4_size))
  
  if os.path.isfile(batch3_in):
    print("Open input file: {} ".format(batch3_in))  
  else:
    print("{} doesn't exist".format(batch3_in))
  file_size = os.path.getsize(batch3_in)
  if file_size < (addr+is_3_size):
    print("size of {} is error".format(batch3_in))
  
  f_in = open(batch3_in, "rb")
  f_in.seek(addr,0)
  read_buf = f_in.read(is_3_size)
  f_out = open(batch3_is_out,"wb")
  f_out.write(read_buf)
  f_out.close()
  f_in.close()
  
  if os.path.isfile(batch4_in):
    print("Open input file: {} ".format(batch4_in))
  else:
    print("{} doesn't exist".format(batch4_in))
  file_size = os.path.getsize(batch4_in)
  if file_size < (addr+is_4_size):
    print("size of {} is error".format(batch4_in))
  
  f_in = open(batch4_in, "rb")
  f_in.seek(addr,0)
  read_buf = f_in.read(is_4_size)
  f_out = open(batch4_is_out,"wb")
  f_out.write(read_buf)
  f_out.close()
  f_in.close()
  
  print("=========end of  generate instr bin==========\n")


