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


import numpy as np
import os
import json

from argparse import ArgumentParser
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
#from utils_tf import dboxes_R34_coco, Encoder, cocoval, MyEncoder

def parse_args():
    parser = ArgumentParser(description="Yolov4 evaluation on COCO")
    parser.add_argument('--output', '-o', type=str,
            default='../../data/result.json',
            help='detection output JSON file path')
    parser.add_argument('--gt-json','-g', type=str,
            default='../../data/coco/instance_5k.json',
            help='path to ground truth json annotations')
    return parser.parse_args()

# This function is from https://github.com/zengarden/light_head_rcnn
def cocoval(detected_json, eval_json):
    eval_gt = COCO(eval_json)

    eval_dt = eval_gt.loadRes(detected_json)
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='bbox')

    # cocoEval.params.imgIds = eval_gt.getImgIds()
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

class MyEncoder(json.JSONEncoder):
   def default(self, obj):
     if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
       np.int16, np.int32, np.int64, np.uint8,
       np.uint16,np.uint32, np.uint64)):
       return int(obj)
     elif isinstance(obj, (np.float_, np.float16, np.float32, 
       np.float64)):
       return float(obj)
     elif isinstance(obj, (np.ndarray,)): # add this line
       return obj.tolist() # add this line
     return json.JSONEncoder.default(self, obj)


if __name__=='__main__':
    args = parse_args()
    cocoval(args.output, args.gt_json)
    
