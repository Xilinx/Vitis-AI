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

args = parse_args()

def gen_list(args):
    l = os.listdir(args.input_dir)
    count = 0
    f = open(args.output_file, 'w')
    for x in l:
        count = count + 1
        f.write('%s %d\n'%("../../data/quantize/images/" + x, count))

if __name__ == "__main__":
   gen_list(args)
