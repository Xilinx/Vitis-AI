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

import argparse
import os.path
import math
import sys
import timeit
import json
import multiprocessing as mp
import numpy as np
from vai.dpuv1.rt import xdnn, xdnn_io
from six import itervalues, iterkeys

def run(pid, args, resultQ):
  fpgaRT = xdnn.XDNNFPGAOp(args)
  xdnnCPUOp = xdnn.XDNNCPUOp(args['weights'])
  args['in_shape'] = tuple((fpgaRT.getBatchSize(),) + tuple(next(itervalues(fpgaRT.getInputDescriptors()))[1:] ))
  fpgaInput = np.empty(args['in_shape'], dtype=np.float32, order='C')
  fpgaOutput = np.empty ((fpgaRT.getBatchSize(), int(args['fpgaoutsz']),), dtype=np.float32, order='C')
  labels = xdnn_io.get_labels(args['labels'])

  img_paths = xdnn_io.getFilePaths(args['images'])
  for j, p in enumerate(img_paths[:fpgaRT.getBatchSize()]):
    fpgaInput[j, ...], _ = xdnn_io.loadImageBlobFromFile(p, args['img_raw_scale'],
                                                            args['img_mean'],
                                                            args['img_input_scale'],
                                                            args['in_shape'][2],
                                                            args['in_shape'][3])

  firstInputName = next(iterkeys(fpgaRT.getInputs()))
  firstOutputName = next(iterkeys(fpgaRT.getOutputs()))
  fpgaRT.exec_async({ firstInputName: fpgaInput },
                    { firstOutputName: fpgaOutput })
  fpgaRT.get_result()

  fcOut = np.empty((fpgaRT.getBatchSize(), args['outsz']), dtype=np.float32, order = 'C')
  xdnnCPUOp.computeFC(fpgaOutput, fcOut)
  softmaxOut = xdnnCPUOp.computeSoftmax(fcOut)
  result = xdnn_io.getClassification(softmaxOut, args['images'], labels);
  resultQ.put((pid, result))

# example for multiple executors
def main():
  args = xdnn_io.processCommandLine()

  # spawn 1 process for each run
  resultQ = mp.Queue()
  procs = []
  for pid, runArgs in enumerate(args['jsoncfg']):
    proc = mp.Process(target=run, args=(pid, runArgs, resultQ,))
    proc.start()
    procs.append(proc)

  # collect results out-of-order
  results = {}
  for p in procs:
    (pid, result) = resultQ.get()
    results[pid] = result

  # print results in order
  for pid, p in enumerate(procs):
    print(results[pid])
    p.join()

if __name__ == '__main__':
  main()
