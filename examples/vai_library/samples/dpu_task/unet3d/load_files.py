"""
Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import numpy as np

folder = ""

if len(sys.argv) < 2:
    folder = "/group/xbjlab/dphi_software/software/workspace/huizhang/accuracy_test_library/unet3d/preprocessed_data/"
else:
    folder = sys.argv[1]

imgs = os.listdir(folder)
n = len(imgs)
a = 0
f =  open('shapes.txt', 'w')
for i in range(n):
  if 'x' in imgs[i]:
    #print(a, imgs[i])
    b = np.load(folder + imgs[i]).shape
    name = imgs[i].split(".")[0]
    name = name.split("_")[1]
    print(name, b)

    for i in range(len(b)):
        name = name+" "+str(b[i])
    name +="\n"
    f.write(name)
f.close()
