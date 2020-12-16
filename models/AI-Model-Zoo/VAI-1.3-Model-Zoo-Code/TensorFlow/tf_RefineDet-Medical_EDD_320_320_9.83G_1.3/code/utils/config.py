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

class priorbox_config(object):
    input_shape = [320, 320]
    feature_shapes = [(40, 40), (20, 20), (10, 10), (5, 5)] 
    min_sizes = [(32,), (64,), (128,), (256,)] 
    max_sizes = [(64,), (128,), (256,), (315,)] 
    aspect_ratios = [(2.,), (2.,), (2.,), (2.,)] 
    steps = [(8, 8), (16, 16), (32, 32), (64, 64)]  
    offset=0.5
    variances = [0.1, 0.1, 0.2, 0.2]

class detect_config(object):
    filter_obj_score = 0.01
    min_size = 0.003 
    keep_topk = 400
    keep_nms_maxnum = 200
    nms_threshold = 0.45
    max_num_bboxes = 200 

