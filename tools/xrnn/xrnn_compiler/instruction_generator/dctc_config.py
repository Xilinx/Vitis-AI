

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

import json
import sys
import os

load_src_reg_0={'dir':1, 'size':32}
load_src_reg_1={'dir':1, 'size':128}
save_dest_reg_0={'dir':1, 'size': 128}
batch4_lines= [341, 69, 5]
batch3_lines= [334, 66, 4]

def save_config(SAVE_PATH, input_size, line_number_4, line_number_3):
    input_temp={}
    for i in range(len(input_size)):
        dir_size=[]
        for load_index, load_size in input_size[i].items():
            dir_size.append([load_size['dir'],load_size['size']])
        input_temp[i] = {'load_src_reg_0':{'dir': dir_size[0][0], 'size':dir_size[0][1]}, 'load_src_reg_1':{'dir': dir_size[1][0], 'size':dir_size[1][1]}, 'save_dest_reg_0':{'dir': dir_size[1][0], 'size':dir_size[1][1]}}
    
    json_write={'layer': input_temp, 'batch4_lines': line_number_4, 'batch3_lines': line_number_3}
    fp_json_write = json.dumps(json_write, indent=1)
    with open(os.path.join(SAVE_PATH,'config.json'), encoding='utf-8', mode='w') as f:
        f.write(fp_json_write)

if __name__ == "__main__":
    save_json(os.path.join(sys.argv[1], 'config.json'))