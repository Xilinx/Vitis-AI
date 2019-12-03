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
import collections
import json
import logging
import multiprocessing as mp
import multiprocessing.pool as mpp
import numpy as np
import time, timeit
import os, sys
import threading
from functools import partial
from vai.dpuv1.rt import xstream, xdnn_io, xdnn
from vai.dpuv1.rt.vitis.python.dpu.runner import Runner

logging.basicConfig(filename='mp_classify_vitis.log', filemode='w', level=logging.DEBUG)

g_nDispatchers = 8
g_nWorkers = 4
g_nQueries = 5000

def chanIdx2Str(idx):
  return "XS_%03d" % idx

def pid2TokenStr():
  return "XSTOKEN_%d_%d" % (os.getpid(), threading.get_ident())

class LRUCache:
  def __init__(self, capacity):
    self.capacity = capacity
    self.cache = collections.OrderedDict()

  def get(self, key):
    try:
      value = self.cache.pop(key)
      self.cache[key] = value
      return value
    except KeyError:
      return None

  def set(self, key, value):
    try:
      self.cache.pop(key)
    except KeyError:
      if len(self.cache) >= self.capacity:
          self.cache.popitem(last=False)
    self.cache[key] = value

class Dispatcher():
  xspub = {}
  xstoken = {}
  inshape = None
  nWorkers = 0
  inBlob = {}
  inBlobCache = LRUCache(512)

  def __init__(self, nDispatchers, nWorkers, inshape):
    self.nWorkers = nWorkers
    Pool = mp.Pool # mpp.ThreadPool
    self.pool = Pool(initializer=Dispatcher._init, processes=nDispatchers,
      initargs=(nWorkers, inshape,))

  @staticmethod
  def _init(nWorkers, inshape):
    token = pid2TokenStr()
    if token not in Dispatcher.xspub:
      Dispatcher.xspub[token] = xstream.Publisher()
      Dispatcher.xstoken[token] = xstream.Subscribe(token)
    Dispatcher.inshape = inshape
    Dispatcher.nWorkers = nWorkers
    Dispatcher.inBlob[token] = np.zeros(tuple(inshape), dtype=np.float32, order='C')

  @staticmethod
  def _run(work):
    try:
      (idx, images, args) = work
      chanIdx = idx % Dispatcher.nWorkers
      token = pid2TokenStr()
      shape = Dispatcher.inshape

      for i, img in enumerate(images):
        cached = Dispatcher.inBlobCache.get(img)
        if cached is None:
          Dispatcher.inBlob[token][i, ...], _ = xdnn_io.loadImageBlobFromFile(img,
            args[0], args[1], args[2], shape[2], shape[3])
          Dispatcher.inBlobCache.set(img, np.copy(Dispatcher.inBlob[token][i]))
        else:
          Dispatcher.inBlob[token][i, ...] = cached

      meta = { 'id': idx, 'from': token, 'shape': shape, 'images': images }
      if idx % 1000 == 0:
        print("Put query %d to objstore" % idx)
        sys.stdout.flush()

      Dispatcher.xspub[token].put_blob(chanIdx2Str(chanIdx),
        Dispatcher.inBlob[token], meta)
      Dispatcher.xstoken[token].get_msg()
    except Exception as e:
      logging.error("Producer exception " + str(e))

  def run(self, work):
    self.pool.map_async(Dispatcher._run, work)

  def __del__(self):
    self.pool.close()
    self.pool.join()

class FpgaMaster():
  def __init__(self, rundir, numFPGAs=1):
    self.hwq = mp.Queue()
    self.hw = mp.Process(target=FpgaMaster.run,
      args=(rundir, numFPGAs, self.hwq,))
    self.hw.start()
    self.inshape = self.hwq.get() # wait to finish acquiring resources

  @staticmethod
  def run(rundir, n, q):
    runners = []
    for i in range(n):
      runners.append(Runner(rundir))
    inTensors = runners[0].get_input_tensors()
    shape = [inTensors[0].dims[i] for i in range(inTensors[0].ndims)]
    q.put(shape) # ready for work
    q.get()  # wait for exit signal

  def __del__(self):
    self.hwq.put(0)
    self.hw.join()

class WorkerPool():
  def __init__(self, rundir, nWorkers, workerArgs):
    self.xspub = xstream.Publisher()
    self.workers = []
    self.wq = mp.Queue()
    for wi in range(nWorkers):
      w = mp.Process(target=WorkerPool.run, args=(rundir, wi, self.wq, workerArgs,))
      w.start()
      self.workers.append(w)
      # wait for worker to be ready before starting next worker
      # because the last worker overwrites the IP programming
      # (Note: this assumes all workers must have the same IP instructions)
      self.wq.get()

  @staticmethod
  def run(rundir, chanIdx, q, args):
    xspub = xstream.Publisher()
    xssub = xstream.Subscribe(chanIdx2Str(chanIdx))
    runner = Runner(rundir)
    inTensors = runner.get_input_tensors()
    outTensors = runner.get_output_tensors()

    q.put(1) # ready for work

    fpgaBlobs = None
    fcOutput = None
    labels = xdnn_io.get_labels(args['labels'])
    xdnnCPUOp = xdnn.XDNNCPUOp("%s/weights.h5" % rundir)
    while True:
      try:
        payload = xssub.get()
        if not payload:
          break
        (meta, buf) = payload

        if fpgaBlobs == None:
          # allocate buffers
          fpgaBlobs = []
          batchsz = meta['shape'][0] # inTensors[0].dims[0]

          for io in [inTensors, outTensors]:
            blobs = []
            for t in io:
              shape = (batchsz,) + tuple([t.dims[i] for i in range(t.ndims)][1:])
              blobs.append(np.empty((shape), dtype=np.float32, order='C'))
            fpgaBlobs.append(blobs)

          fcOutput = np.empty((batchsz, args['outsz'],), dtype=np.float32, order='C')

        fpgaInput = fpgaBlobs[0][0]
        assert(tuple(meta['shape']) == fpgaInput.shape)
        data = np.frombuffer(buf, dtype=np.float32).reshape(fpgaInput.shape)
        np.copyto(fpgaInput, data)

        jid = runner.execute_async(fpgaBlobs[0], fpgaBlobs[1])
        runner.wait(jid)

        xdnnCPUOp.computeFC(fpgaBlobs[1][0], fcOutput)
        softmaxOut = xdnnCPUOp.computeSoftmax(fcOutput)
        xdnn_io.printClassification(softmaxOut, meta['images'], labels)
        sys.stdout.flush()

        if meta['id'] % 1000 == 0:
          print("Recvd query %d" % meta['id'])
          sys.stdout.flush()

        del data
        del buf
        del payload

        xspub.send(meta['from'], "success")

      except Exception as e:
        logging.error("Worker exception " + str(e))

  def __del__(self):
    # close workers
    for wi, w in enumerate(self.workers):
      self.xspub.end(chanIdx2Str(wi))
      w.join();

def main():
  args = xdnn_io.processCommandLine()
  images = xdnn_io.getFilePaths(args['images'])

  # start comms
  xserver = xstream.Server()

  # acquire resources
  fmaster = FpgaMaster(args['vitis_rundir'])
  inshape = list(fmaster.inshape)
  if args['batch_sz'] != -1:
    inshape[0] = args['batch_sz'] # update batch size

  # spawn dispatchers
  dispatcher = Dispatcher(g_nDispatchers, g_nWorkers, inshape)

  # spawn workers
  workers = WorkerPool(args['vitis_rundir']+"_worker", g_nWorkers, args)

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

  # cleanup
  del workers
  del fmaster
  del xserver

if __name__ == '__main__':
  main()
