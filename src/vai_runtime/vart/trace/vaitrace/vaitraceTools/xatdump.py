#!/usr/bin/python3

# Copyright 2022-2023 Advanced Micro Devices Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pickle
import gzip
import os
import sys

IN = sys.argv[1]

try:
    TYPE = sys.argv[2]
except:
    TYPE = None
try:
    OUT = sys.argv[3]
except:
    OUT = None

with gzip.open(IN, 'rb') as xatRaw:
    dataRaw = pickle.load(xatRaw)

    for k in dataRaw.keys():
        if TYPE == None:
            print("Usage: xatDump [input] [key | all]")
            print("Available Keys: ", dataRaw.keys())
            exit(0)
        if TYPE == 'all':
            pass
        else:
            if k.lower().find(TYPE.lower()) < 0:
                continue
        if len(dataRaw[k]) == 0:
            continue

        if type(dataRaw[k]) == list:
            print(*dataRaw[k])
        else:
            print(dataRaw[k])
