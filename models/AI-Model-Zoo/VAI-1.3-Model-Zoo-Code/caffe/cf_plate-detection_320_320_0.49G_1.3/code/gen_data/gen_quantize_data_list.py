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


#coding=utf-8
import os
import sys
import numpy as np
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="generate quantize list script")
    parser.add_argument('--input_dir', type=str,
            default='./data/quantize/images/',
            help='The source directory to be looked for')
    parser.add_argument('--output_file', type=str,
            default='./data/quantize/quant.txt',
            help='The images list file')
    return parser.parse_args()

count = 0
args = parse_args()

def gen_list(path):
    for root, dirs, files in os.walk(path):
        for sub_dir in dirs:
            gen_list(sub_dir)
        f = open(args.output_file, 'w')
        for e in files:
            global count
            count = count + 1
            f.write('%s %d\n'%("../." + root + "/" + e, count))


if __name__ == "__main__":
   gen_list(args.input_dir)
