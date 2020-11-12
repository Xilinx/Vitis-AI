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


import cv2
import os
import argparse
import numpy as np
from PIL import Image
#from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Executor
def get_arguments():
    """Parse all the arguments.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Create lists")
    parser.add_argument("--root_dir", type=str, default='./data/cityscapes',
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--images_dir", type=str, default='leftImg8bit',
                        help="Path to the directory containing the dataset images.")
    parser.add_argument("--annotations_dir", type=str, default='gtFine',
                        help="Path to the directory containing the dataset annotations.")
    parser.add_argument("--lists_dir", type=str, default='lists',
                        help="where to save the list file.")
    parser.add_argument("--image_suffix", type=str, default='_leftImg8bit.png',
                        help="image file suffix")
    parser.add_argument("--annotation_suffix", type=str, default='_gtFine_trainids.png',
                        help="annotation file suffix")
    
    return parser.parse_args()


    args = get_arguments()

def relabel(annotations_dir):
    ignore_label = 19
    id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                 3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                 7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                 14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                 18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                 28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
    for root, dirs, filenames in os.walk(annotations_dir):
        for name in filenames:
            parts = name.split('_')
            if parts[-1] != 'labelIds.png':
                continue
            mask = Image.open(os.path.join(root, name))
            mask = np.array(mask)
            mask_copy = mask.copy()
            for k, v in id_to_trainid.items():
                mask_copy[mask == k] = v
            relabel_mask= Image.fromarray(mask_copy.astype(np.uint8))
            out_path = os.path.join(root, name[:-12]+'trainids.png')
            relabel_mask.save(out_path)

def create_list(root_dir, images_dir, annotations_dir, lists_dir, split, image_suffix, annotation_suffix):
    assert split in ['train', 'val']
    image_path = os.path.join(images_dir, split)
    annotation_path = os.path.join(annotations_dir, split)
    image_lists, annotation_lists= [], []
    categories = os.listdir(os.path.join(root_dir, image_path))
    for c in categories:
        c_items = [name.split(image_suffix)[0] for name in os.listdir(os.path.join(root_dir, image_path, c))]
        for it in c_items:
            image_name = it
            image_file = os.path.join(image_path, c, it + image_suffix)
            annotation_file = os.path.join(annotation_path, c, it + annotation_suffix)
            image_lists.append(image_file)
            annotation_lists.append(annotation_file)
            if not os.path.exists(os.path.join(root_dir, image_file)):
                print(image_file, "not exist")
            if not os.path.exists(os.path.join(root_dir, annotation_file)):
                print(annotation_file, "not exist")

    assert len(image_lists) == len(annotation_lists)

if __name__=='__main__':

    args = get_arguments()    
    root_dir = args.root_dir
    images_dir = args.images_dir  
    annotations_dir = args.annotations_dir  
    lists_dir = args.lists_dir
    image_suffix = args.image_suffix
    annotation_suffix = args.annotation_suffix

    print(" =====> relabel annotation file to [0-18]+[255]")
    relabel(os.path.join(root_dir, annotations_dir))
    #print(" =====> creat list files as format: [ImagePath LabelPath]")
    #for split in ['train','val']:
    #    create_list(root_dir, images_dir, annotations_dir, lists_dir, split, image_suffix, annotation_suffix)



