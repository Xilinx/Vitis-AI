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

import sys
import time
import glob

from apps.aks.libs import aks

def usage(exe):
  print("[INFO] Usage: ")
  print("[INFO] ---------------------- ")
  print("[INFO] ", exe, " <Image Directory Path>")

def main(imageDirectory):
  
  fileExtension = ('*.jpg', '*.JPEG', '*.png')
  images = []
  for ext in fileExtension:
    images.extend(glob.glob(imageDirectory + '/' + ext))

  kernelDir = "kernel_zoo"
  graphJson = "graph_zoo/graph_std_yolov2_608.json"
  graphName = "std_yolov2_608"

  sysMan = aks.SysManager()
  sysMan.loadKernels(kernelDir)
  sysMan.loadGraphs(graphJson)
  graph = sysMan.getGraph(graphName)

  print("[INFO] Starting enqueue... ")
  print("[INFO] Running", len(images), "images")
  t0 = time.time()
  for i, img in enumerate(images):
    sysMan.enqueueJob(graph, img)

  sysMan.waitForAllResults()
  t1 = time.time()
  print("[INFO] Overall FPS : ", len(images)/(t1-t0))

  sysMan.report(graph)

if __name__ == "__main__":
  if (len(sys.argv) != 2):
    print("[ERROR] Invalid Usage!")
    usage(sys.argv[0])
  else:
    main(sys.argv[1])
