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


import csv
import os
import sys
from collections import namedtuple


try:
    CSV_PATH = sys.argv[1]
except:
    CSV_PATH = "./test_dpu_mc.csv"

csv_f = open(CSV_PATH, "rt")
csv_reader = csv.reader(csv_f)
csv_header = next(csv_reader)

analysis_list = []
XModelRaw = namedtuple("xmodel_raw", csv_header)

# prepare data
for m in csv_reader:
    analysis_list.append(XModelRaw(*m))

result_list = []
XModelAnaRes = namedtuple(
    "xmodel_ana", ["raw", "total_read", "total_acc", "comp_density"])
# analysis
for m in analysis_list:
    total_read = int(m.load_img_size) + int(m.load_para_size)
    total_acc = int(m.save_size) + total_read
    workload = int(m.workload)
    den = workload / total_acc

    result_list.append(XModelAnaRes(m, total_read, total_acc, den))

# print result
result_list.sort(key=lambda x: x.comp_density)
for m in result_list:
    print("%-s:%-s:   %.3f" %
          (m.raw.xmodel, m.raw.subgraph_name, m.comp_density))

csv_f.close()
