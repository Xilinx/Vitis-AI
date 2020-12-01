

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
from ctypes import *

def gen_bin(layer_name_list, line_number_3, line_number_4, SAVE_PATH):

    print("=========begin to generate ddr bin==========\n")
    lib = cdll.LoadLibrary("bin_gen.so")
    ddr_init = os.path.join(SAVE_PATH, "ddr_init_orign.txt")
    instr_f3 = os.path.join(SAVE_PATH, "instr_ac_3.txt")    
    ddr_bin3 = os.path.join(SAVE_PATH, "ddr_bin_3k")
    

    layer_num = len(layer_name_list)
    line_dim3 = len(line_number_3)    
    line_3 = (c_uint*line_dim3)()    
    for i in range(0, len(line_number_3)):
        line_3[i] = line_number_3[i]

    lib.create_bin.argtypes = c_uint, c_uint, POINTER(c_uint), c_char_p, c_char_p, c_char_p
    lib.create_bin(c_uint(layer_num), c_uint(3), line_3, c_char_p(ddr_init.encode("utf-8")), c_char_p(instr_f3.encode("utf-8")), \
                 c_char_p(ddr_bin3.encode("utf-8")))


    instr_f4 = os.path.join(SAVE_PATH, "instr_ac_4.txt")
    ddr_bin4 = os.path.join(SAVE_PATH, "ddr_bin_4k")
    
    line_dim4 = len(line_number_4)    
    line_4 = (c_uint*line_dim4)()
    for i in range(0, len(line_number_4)):
        line_4[i] = line_number_4[i]


    lib.create_bin(c_uint(layer_num), c_uint(4), line_4, c_char_p(ddr_init.encode("utf-8")), c_char_p(instr_f4.encode("utf-8")), \
                 c_char_p(ddr_bin4.encode("utf-8")))


    print("=========end of  generating ddr bin==========\n")

