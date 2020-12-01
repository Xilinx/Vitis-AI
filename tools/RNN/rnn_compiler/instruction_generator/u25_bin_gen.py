

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

def gen_bin(layer_name_list, SAVE_PATH):

    print("=========begin to generate ddr bin==========\n")
    lib = cdll.LoadLibrary("u25_bin_gen.so")

    ddr_init = os.path.join(SAVE_PATH, "ddr_init_orign.txt")
    instr_f = os.path.join(SAVE_PATH, "instr_ac.txt")    
    ddr_bin = os.path.join(SAVE_PATH, "u25_ddr_bin")


    if len(layer_name_list)==1:
        model = "sent"
    else:
        model = "openie"

    lib.create_bin(c_char_p(ddr_init.encode("utf-8")), c_char_p(instr_f.encode("utf-8")), \
                   c_char_p(ddr_bin.encode("utf-8")), c_char_p(model.encode("utf-8")), c_uint(0x7000000))

    print("=========end of  generating ddr bin==========\n")
