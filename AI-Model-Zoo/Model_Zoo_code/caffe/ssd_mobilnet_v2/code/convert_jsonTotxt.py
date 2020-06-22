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


import string
import os
import argparse
import json

parser = argparse.ArgumentParser(
    description='Convert bdd100k json file to txt file')
parser.add_argument('--json_file_path',
                    default=None, type=str,
                    help='json file path to open')
parser.add_argument('--txt_file_path',
                    default=None, type=str,
                    help='txt file path to open')
args = parser.parse_args()
classes = ['bike','bus','car','motor','person','rider','traffic light','traffic sign','train','truck']

def convertJsonTotxt(filename ,json_file_path, txt_file_path):
    out_file = open(txt_file_path + "/" + filename.split('/n')[0].split('.')[0] + ".txt" ,'w')
    json_line = json.load(open(json_file_path  + "/"  + filename , "r"))
    objects = json_line['frames'][0]['objects']
    image_name = json_line['name']
    for m in range(len(objects)):
        category = objects[m]['category']
        if category in classes:
            cls = category.replace('traffic ','')  
            xmin = objects[m]['box2d']['x1']
            ymin = objects[m]['box2d']['y1']
            xmax = objects[m]['box2d']['x2']
            ymax = objects[m]['box2d']['y2']
            bbox = [image_name,' ',cls,' ',xmin,'' ,ymin,' ',xmax,' ',ymax]
            for a in bbox:
                out_file.write(str(a))
            out_file.write('\n')
 

if __name__ == '__main__':
    json_file_path=args.json_file_path
    txt_file_path=args.txt_file_path

    for filename in os.listdir(json_file_path):
        convertJsonTotxt(filename, json_file_path, txt_file_path)
