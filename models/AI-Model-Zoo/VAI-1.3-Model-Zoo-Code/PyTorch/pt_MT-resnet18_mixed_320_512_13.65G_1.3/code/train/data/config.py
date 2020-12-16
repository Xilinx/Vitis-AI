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

# PART OF THIS FILE AT ALL TIMES.

import os.path
# gets home dir cross platform
HOME = os.path.expanduser("~")

MEANS = (104, 117, 123)
#MEANS = (72,83,73)

# CONFIGS
solver = {
    'k1': 8,
    'k2': 8,
    'act_clip_val': 8,
    'warmup': False,
    'det_classes': 4,
    'seg_classes': 16,
    'lr_steps': (5000, 7000,9000),
    #'lr_steps': (5, 10),
    'max_iter': 10000,
    'feature_maps': [(80,128), (40,64), (20,32), (10,16), (5,8), (3,6), (1,4)],
    'resize': (320,512),
    'steps': [4, 8, 16, 32, 64, 128, 256],
    'min_sizes': [10, 30, 60, 100, 160, 220, 280],
    'max_sizes': [30, 60, 100, 160, 220, 280, 340],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2]],
    'variance': [0.1, 0.2],
    'clip': False,
}


