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


import os
import argparse

def gen_sample_list(idx_list, dst_list):
    with open(idx_list) as fr:
        lines = fr.readlines()
    with open(dst_list, 'w') as fw:
        for line in lines:
            idx = line.strip().zfill(12)
            name = 'COCO_val2014_' + idx
            fw.write(name + '\n')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script to generate the validation list')
    parser.add_argument('-idx_list', default='code/test/dataset_tools/mscoco_minival_ids.txt')
    parser.add_argument('-dst_list', default='data/coco2014_minival_8059/minival2014_8059.txt')
    args = parser.parse_args()

    dst_dir, dst_file = os.path.split(args.dst_list)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    gen_sample_list(args.idx_list, args.dst_list)
    print("Validation list is saved as {}".format(args.dst_list))
