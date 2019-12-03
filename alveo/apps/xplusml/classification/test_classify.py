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
from six import itervalues, iteritems
from ctypes import *
import numpy as np

import os, sys
from vai.dpuv1.rt import xdnn, xdnn_io
from vai.dpuv1.rt.vitis.python.dpu.runner import Runner
import xplusml

import multiprocessing as mp
import ctypes


def pre_process(q,args):

  xclbin_p=str(args['xclbin']+"/xdnn_v3_96x16_2pe_8b_9mb_bank03.xclbin")
  kernelName_p="pp_pipeline_accel"
  deviceIdx_p=args['deviceid']
  fpga_pp = xplusml.PreProcess(xclbin_p,kernelName_p,deviceIdx_p)
  batch_sz = args['batch_sz']
  img_paths = xdnn_io.getFilePaths(args['images'])
  print("Pre-processing handle created. Populating Queue")
  for i in range(0, len(img_paths), batch_sz):
    for j, p in enumerate(img_paths[i:i + batch_sz]):
      arr, ht = fpga_pp.preprocess_input(p)
      q.put(arr)
      print("Queue populated")


def process_xdnn(q,args):
  runner = Runner(args['vitis_rundir'])
  inTensors = runner.get_input_tensors()
  outTensors = runner.get_output_tensors()
  batch_sz = args['batch_sz']
  if batch_sz == -1:
    # use Runner's suggested batch size
    batch_sz = inTensors[0].dims[0]

  if args['golden']:
    goldenMap = xdnn_io.getGoldenMap(args['golden'])
    top5Count = 0
    top1Count = 0

  fpgaBlobs = []
  for io in [inTensors, outTensors]:
    blobs = []
    for t in io:
      shape = (batch_sz,) + tuple([t.dims[i] for i in range(t.ndims)][1:])
      blobs.append(np.empty((shape), dtype=np.float32, order='C'))
    fpgaBlobs.append(blobs)

  img_paths = xdnn_io.getFilePaths(args['images'])
  labels = xdnn_io.get_labels(args['labels'])
  xdnnCPUOp = xdnn.XDNNCPUOp("%s/weights.h5" % args['vitis_rundir'])
  fcOutput = np.empty((batch_sz, args['outsz'],), dtype=np.float32, order='C')

  fpgaInput = fpgaBlobs[0][0]
  for i in range(0, len(img_paths), batch_sz):
    pl = []
    # fill tensor input data from image file
    for j, p in enumerate(img_paths[i:i + batch_sz]):

      img, _ = q.get(), None
      pl.append(p)
      np.copyto(fpgaInput[j], img)

    jid = runner.execute_async(fpgaBlobs[0], fpgaBlobs[1])
    runner.wait(jid)

    xdnnCPUOp.computeFC(fpgaBlobs[1][0], fcOutput)
    softmaxOut = xdnnCPUOp.computeSoftmax(fcOutput)
    if args['golden']:
      for j,p in enumerate(img_paths[i:i + batch_sz]):
        top1Count += xdnn_io.isTopK(softmaxOut[j], goldenMap, p, labels, 1)
        top5Count += xdnn_io.isTopK(softmaxOut[j], goldenMap, p, labels, 5)
    else:
      xdnn_io.printClassification(softmaxOut, pl, labels)

  if args['golden']:
    print ( ("\nAverage accuracy (n=%d) Top-1: %.1f%%, Top-5: %.1f%%\n") % (len(img_paths), float(top1Count)/float(len(img_paths))*100., float(top5Count)/float(len(img_paths))*100.) )

if __name__ == '__main__':

        args = xdnn_io.processCommandLine()
		#Create a queue for passing the pre-processed data
        q = mp.Queue()
		#Creating a process to run HW pre-processing kernel
        p_preprocess = mp.Process(target=pre_process,args=(q,args))
		#Process to run XDNN
        p_xdnn = mp.Process(target=process_xdnn,args=(q,args))

        p_preprocess.start()
        p_xdnn.start()
        p_preprocess.join()
        p_xdnn.join()
