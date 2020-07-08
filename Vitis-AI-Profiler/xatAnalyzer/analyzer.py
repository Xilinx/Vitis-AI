#!/usr/bin/python3
# -*- coding: UTF-8 -*-

# Copyright 2019 Xilinx Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import re
import sys
import csv
import pickle
import gzip
import json
import parser

IN = sys.argv[1]
OUT = sys.argv[2]

xat = gzip.open(IN, 'rb')
raw = pickle.load(xat)
xat.close()

print(raw.keys())

globalOption = raw.get('cmd', {})
out = parser.parse(raw, globalOption)
ts, te = parser.timelineUtil.timelineAdj(out, globalOption)
out.update({'TIME-start': [ts], 'TIME-end': [te]})

"""
If work in 'debug' mode, FPS will be meaningless,
because profiling overhead is significant
"""

if globalOption.get('runmode') == "debug":
    out.pop('FPS-0')
    pass

with open(OUT, 'wt+') as f:
    f.write(json.dumps(out, indent=1, sort_keys=True))
