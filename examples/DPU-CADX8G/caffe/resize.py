#!/usr/bin/env python
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

from __future__ import print_function

import os,sys,cv2

def resize(dirpath,width,height):

  if not width:
    sys.exit(1)
  if not height:
    sys.exit(1)

  for _, _, files in os.walk(dirpath):
    for f in files:
      if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".JPG") or f.endswith(".JPEG"):
        fpath = dirpath + "/" + f
        img = cv2.imread(fpath)
        if img is None:
          print ("CV2 Failed to load %s" % f)
          continue
        img = cv2.resize(img,(int(width),int(height)))
        cv2.imwrite(fpath,img)

if __name__ == "__main__":
  # Usage:
  # python resize.py /path/to/images width height
  # i.e.
  # python resize.py ~/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min 256 256
  resize(sys.argv[1],sys.argv[2],sys.argv[3])
