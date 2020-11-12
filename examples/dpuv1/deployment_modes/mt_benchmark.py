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
#
# multi-threaded multi-FPGA max performance stress test
#
from __future__ import print_function
from six import itervalues,iterkeys

from queue import Queue
import json
import logging
import os, sys
import timeit
import numpy as np
import multiprocessing as mp
import ctypes
import signal
import threading
from vai.dpuv1.rt import xdnn, xdnn_io
from vai.dpuv1.rt.vitis.python.dpu.runner import Runner
import time

logging.basicConfig(filename='mt_benchmark.log', filemode='w', level=logging.DEBUG)

class Dispatcher():
  def __init__(self, rundir, nFPGA, nDispatchers, batchsz=-1):
    # update meta.json with nFPGA
    meta = {}
    with open("%s/meta.json" % rundir) as f:
      meta = json.load(f)
      meta['num_fpga'] = nFPGA
      if 'publish_id' in meta:
        del meta['publish_id']
      if 'subscribe_id' in meta:
        del meta['subscribe_id']
    with open("%s/meta.json" % rundir, "w") as f:
      json.dump(meta, f)

    # acquire FPGA
    runner = Runner(rundir)
    inTensors = runner.get_input_tensors()
    outTensors = runner.get_output_tensors()
    inshape = [inTensors[0].dims[i] for i in range(inTensors[0].ndims)]
    outshape = [outTensors[0].dims[i] for i in range(outTensors[0].ndims)]
    if batchsz != -1:
      inshape[0] = batchsz # update batch size
      outshape[0] = batchsz # update batch size

    Dispatcher.runner = runner
    Dispatcher.inTensors = inTensors
    Dispatcher.outTensors = outTensors
    Dispatcher.inshape = inshape
    Dispatcher.outshape = outshape

    self.q = Queue(maxsize=nDispatchers*4)
    self.workers = []
    for i in range(nDispatchers):
      sys.stdout.flush()
      worker = threading.Thread(target=self._run, args=(self.q, inshape, outshape,))
      self.workers.append(worker)
      worker.start()

  @staticmethod
  def _run(q, inshape, outshape):
    inblob = [np.empty((inshape), dtype=np.float32, order='C')]
    outblob = [np.empty((outshape), dtype=np.float32, order='C')]

    while True:
      work = q.get()
      if work is None:
        q.task_done()
        break

      jid = Dispatcher.runner.execute_async(inblob, outblob)
      Dispatcher.runner.wait(jid)

      q.task_done()

  def run(self, work):
    for w in work:
      self.q.put(w)

  def __del__(self):
    for _ in self.workers:
      self.q.put(None)
    for w in self.workers:
      w.join()

g_nDispatchers = 16
g_nQueries = 5000
g_nFPGA = 1
def main(args=None):
  os.environ['LS_BIND_NOW'] = "1"
  args = xdnn_io.processCommandLine()
  images = xdnn_io.getFilePaths(args['images'])

  # spawn dispatcher
  dispatcher = Dispatcher(args['vitis_rundir'],
    g_nFPGA, g_nDispatchers, args['batch_sz'])
  inshape = dispatcher.inshape

  # send work to system
  work = []
  for qIdx in range(g_nQueries):
    idx = qIdx * inshape[0]
    workBatch = [images[(idx+i) % len(images)] for i in range(inshape[0])]
    work.append((qIdx, workBatch,
      (args['img_raw_scale'], args['img_mean'], args['img_input_scale'])))

  startTime = timeit.default_timer()
  dispatcher.run(work)
  del dispatcher
  t = timeit.default_timer() - startTime

  print("Queries: %d, Elapsed: %.2fs, QPS: %.2f, FPS: %.2f" \
    % (g_nQueries, t, g_nQueries / t, g_nQueries * inshape[0] / t))
  sys.stdout.flush()

if __name__ == '__main__':
  main()
