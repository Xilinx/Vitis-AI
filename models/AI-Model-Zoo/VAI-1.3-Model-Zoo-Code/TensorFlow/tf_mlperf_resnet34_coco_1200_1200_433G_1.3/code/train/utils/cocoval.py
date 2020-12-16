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

# encoding: utf-8
"""
@author: jemmy li
@contact: zengarden2009@gmail.com
"""

import argparse
import os, sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from IPython import embed

def cocoval(detected_json, eval_json):
    eval_gt = COCO(eval_json)
    eval_dt = eval_gt.loadRes(detected_json)
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json', type=str, help='json to eval')
    args = parser.parse_args()
