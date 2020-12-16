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
import sys
import tensorflow as tf
import cv2
import numpy
import json

from utils.cocoval import cocoval
import utils.coco_eval_cfg as cfg

def get_labelmap(labelmap_path):
    lines = open(labelmap_path).readlines()
    lines = list(map(lambda x:int(x.strip()),lines))
    lines.insert(0,0)
    return lines

def eval_on_coco(det_res, cfg=cfg):
    coco_records = []
    labelmap = get_labelmap(cfg.coco_labelmap_path)
    for cls_key in det_res.keys():
        for ind, pred in enumerate(det_res[cls_key]):
            record = {}
            image_name = pred[0]
            score = pred[1]
            xmin = pred[2]
            ymin = pred[3]
            xmax = pred[4]
            ymax = pred[5]
            class_id = cls_key
            record['image_id'] = int(image_name.split('_')[-1])
            record['category_id'] = labelmap[class_id]
            record['score'] = score
            record['bbox'] = [xmin, ymin, xmax - xmin + 1, ymax - ymin + 1]
            if score < cfg.score_threshold:
              continue 
            coco_records.append(record)
    with open(cfg.det_file, 'w') as f_det:                       
      f_det.write(json.dumps(coco_records, cls=MyEncoder))
    return cocoval(cfg.det_file, cfg.gt_file)

class MyEncoder(json.JSONEncoder):
   def default(self, obj):
     if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
       numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
       numpy.uint16,numpy.uint32, numpy.uint64)):
       return int(obj)
     elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32, 
       numpy.float64)):
       return float(obj)
     elif isinstance(obj, (numpy.ndarray,)):
       return obj.tolist()
     return json.JSONEncoder.default(self, obj)

if __name__ == '__main__':
    cocoval(cfg.det_file, cfg.gt_file)
