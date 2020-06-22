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

import sys, os
import glob
import time
import threading

from apps.aks.libs import aks

def usage(exe):
  print("[INFO] Usage: ")
  print("[INFO] ---------------------- ")
  print("[INFO] ", exe, " <img-dir-for-googlenet> <image-dir-for-resnet50> ")


def enqJobThread (name, graph, images):
  # Get AKS Sys Manager
  sysMan = aks.SysManager()
  print ("[INFO] Starting Enqueue:", name)
  for img in images:
    sysMan.enqueueJob(graph, img)

def main(imageDirectory, graphs):
  
  fileExtension = ('*.jpg', '*.JPEG', '*.png')

  kernelDir = "kernel_zoo"

  sysMan = aks.SysManager()
  sysMan.loadKernels(kernelDir)

  lgraphs = {}
  images = {}
  # Load graphs
  for graphName, graphJson in graphs.items():
    sysMan.loadGraphs(graphJson)
    lgraphs[graphName] = sysMan.getGraph(graphName)

  images = {}
  for graphName in lgraphs.keys():
    images[graphName] = []
    for ext in fileExtension:
      images[graphName].extend(glob.glob(imageDirectory[graphName] + '/' + ext))

  pushThreads = []
  sysMan.resetTimer()
  t0 = time.time()
  for name, gr in lgraphs.items():
    th = threading.Thread(target=enqJobThread, args=(name, gr, images[name],))
    th.start()
    pushThreads.append(th)

  for th in pushThreads:
    th.join()

  sysMan.waitForAllResults()
  t1 = time.time()
  print("\n[INFO] Overall FPS:", len(images) * 2 / (t1-t0))

  for name, gr in lgraphs.items():
    print("\n[INFO] Graph:", name)
    sysMan.report(gr)

  print("")
  # Destroy SysMan
  sysMan.clear()

if __name__ == "__main__":
  if (len(sys.argv) != 3):
    print("[ERROR] Invalid Usage!")
    usage(sys.argv[0])
    exit(1)

  if not os.path.isdir(sys.argv[1]):
      print("[ERROR] No such directory:", sys.argv[1])
      usage(sys.argv[0])
      exit(1)
  if not os.path.isdir(sys.argv[2]):
      print("[ERROR] No such directory:", sys.argv[2])
      usage(sys.argv[0])
      exit(1)

  # Get images
  imageDirectory = {}
  imageDirectory['googlenet_no_runner'] = sys.argv[1] 
  imageDirectory['resnet50_no_runner'] = sys.argv[2]

  # GoogleNet and TinyYolo-v3 graphs
  graphs = {}
  graphs['googlenet_no_runner'] = 'graph_zoo/graph_googlenet_no_runner.json'
  graphs['resnet50_no_runner'] = 'graph_zoo/graph_resnet50_no_runner.json'

  # Process graphs
  main(imageDirectory, graphs)

