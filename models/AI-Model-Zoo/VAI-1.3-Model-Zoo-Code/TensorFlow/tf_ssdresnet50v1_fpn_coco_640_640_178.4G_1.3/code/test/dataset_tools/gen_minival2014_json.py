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
import json
import argparse


def gen_sample_gt_json(old_json_file, new_json_file, filter_file):

    filter_list = []
    with open(filter_file, "r") as f_fil:
        lines = f_fil.readlines()
        for line in lines:
            filter_list.append(line.strip())
    
    with open(old_json_file, "r") as f_old:
        old_json_str = f_old.read()
    parse_object = json.loads(old_json_str)
    
    # print(parse_object.keys())
    new_info = parse_object['info']
    new_license = parse_object['licenses']
    new_categories = parse_object['categories']
    new_annotations = parse_object['annotations']
    new_images = parse_object['images']
    
    index = []
    # print(type(filter_list[0]))
    # print(len(parse_object['images']))
    for i in range(len(parse_object['images'])):
        if str(parse_object['images'][i]['id']) in filter_list:
            index.append(i)
    
    new_data = {}
    new_images = [new_images[ind] for ind in index]
    
    index = []
    # print(len(parse_object['annotations']))
    # print(parse_object['annotations'][0])
    for i in range(len(parse_object['annotations'])):
        if str(parse_object['annotations'][i]['image_id']) in filter_list:
            index.append(i)
    new_annotations = [new_annotations[ind] for ind in index]
    
    new_data['info'] = new_info
    new_data['licenses'] = new_license
    new_data['categories'] = new_categories
    new_data['annotations'] = new_annotations
    new_data['images'] = new_images
    with open(new_json_file, 'w') as f_det:
        f_det.write(json.dumps(new_data))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script to generate the validation groundtruth json')
    parser.add_argument('-old_json_file', default='data/annotations/instances_val2014.json')
    parser.add_argument('-new_json_file', default='data/coco2014_minival_8059/minival2014_8059.json')
    parser.add_argument('-filter_file', default='code/test/dataset_tools/mscoco_minival_ids.txt')
    args = parser.parse_args()

    dst_dir, dst_file = os.path.split(args.new_json_file)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    gen_sample_gt_json(args.old_json_file, args.new_json_file, args.filter_file)
    print("Validation groundtruth is saved as {}".format(args.new_json_file))
