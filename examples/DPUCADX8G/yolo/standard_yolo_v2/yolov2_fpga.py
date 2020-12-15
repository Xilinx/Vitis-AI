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
import ctypes
import numpy as np
import multiprocessing as mp
import threading
import time, timeit
from vai.dpuv1.rt import xdnn, xdnn_io
from vai.dpuv1.rt import xstream
from vai.dpuv1.rt.xsnodes.base import XstreamNode
import logging
from six import itervalues, iterkeys, next

logger = logging.getLogger(__name__)

def ingest_worker(qin, qout, inshape, inbufs):
  # fetch remote data into local buffer
  # (we want to keep a static set of buffers because they
  #  are mapped to FPGA DDR memory)
  pclient = xstream.Base()

  while True:
    try:
      payload = qin.get()
      if payload is None:
        break

      (obj_ids, inbufIdxs, sId) = payload
      metas = []
      for i,inbufIdx in enumerate(inbufIdxs):
        (meta, buf) = pclient.obj_get(obj_ids[i])
        metas.append(meta)
        npremote_view = None

        try:
          npremote_view = np.frombuffer(buf, dtype=np.float32).reshape(inshape)
          nplocal_view = np.frombuffer(inbufs[inbufIdx].get_obj(),
            dtype=np.float32).reshape(inshape)

          np.copyto(nplocal_view, npremote_view)
        except:
          # previous node gave us a bad buffer, ignore
          pass

        del npremote_view
        del buf

      qout.put((metas, inbufIdxs, sId))
    except Exception:
      logger.exception("fpga ingest worker error")

  qout.put(None)

class Node(XstreamNode):
  def initialize(self, args):
    # make sure to set up subscribe sockets first, so we don't miss messages
    self.sub_0 = self.get_sub(0)
    self.sub_1 = self.get_sub(1)

    self._compJson = xdnn.CompilerJsonParser(args['netcfg'])
    self._fpgaRT = xdnn.XDNNFPGAOp(args)

    self._numStreams = args['numstream']
    # allocate twice as many buffers than streams here because...
    # we know when 'a' stream/buffer completes, but we don't keep track
    # of exactly which stream/buffer is freed.
    # using double the buffers ensures that we are never clobbering
    # existing streams/buffers
    self._numStreamBuffers = self._numStreams * 2
    self._numStreamsActive = 0
    self._currStreamIdx = 0
    self._bsz = args['batch_sz']

    self._inputBuffers = []
    self._outputBuffers = []
    self._firstInputShape = next(itervalues(self._compJson.getInputs()))
    # firstOutputShape = next(itervalues(self._compJson.getOutputs()))
    outputShapes = list(itervalues(self._compJson.getOutputs()))
    outputNames =  list(iterkeys(self._compJson.getOutputs()))
    
    for si in range(self._bsz * self._numStreamBuffers):
      self._inputBuffers.append(mp.Array(ctypes.c_float,
        np.prod(tuple(self._firstInputShape)).tolist()))
      
      bufs = []
      for outputShape in outputShapes:
        bufs.append(np.empty((self._bsz,) + tuple (outputShape[1:]),
          dtype=np.float32, order='C'))
      self._outputBuffers.append(bufs)

    # print("outputBuffer : ", [len(item) for item in self._outputBuffers])

    # Pipeline:
    # 1) ingest
    #    collect individual requests into 1 batch for 1 stream
    # 2) ingest_worker(s)
    #    copy individual object store blobs into local buffers for 1 stream
    # 3) loop
    #    submit fpga job
    # 4) wait
    #    wait for fpga job

    self._qingest = mp.Queue(maxsize=len(self._inputBuffers))
    self._qfpga = mp.Queue(maxsize=len(self._inputBuffers))
    # spawn ingest_workers to copy remote buffers to local buffer
    self._ingestWorkers = []
    for pi in range(args['numprepproc']):
      p = mp.Process(target=ingest_worker,
        args=(self._qingest, self._qfpga,
          self._firstInputShape, self._inputBuffers,))
      p.start()
      self._ingestWorkers.append(p)

    # ingest thread dispatches incoming work to ingest_workers
    self._ingestThread = threading.Thread(target=self.ingest,
      args=(self._qingest, self._qfpga))
    self._ingestThread.start()

    # wait thread collects completed FPGA results and sends forward
    self._qwait = mp.Queue(maxsize=len(self._inputBuffers))
    self._waitThread = threading.Thread(target=self.wait,
      args=(self._qwait, outputNames,))
    self._waitThread.start()

    print("Starting FPGA loop")

    self.run()

  def get_next_stream_id(self, sub, pub):
    sId = self._currStreamIdx
    self._currStreamIdx = (self._currStreamIdx + 1) % self._numStreamBuffers
    if self._numStreamsActive >= self._numStreams:
      # we've reach max streams allowed, wait for a stream to finish
      for i in range(self._bsz):
        self.get(1, sub, pub)
      self._numStreamsActive -= 1
    self._numStreamsActive += 1
    return sId

  def ingest(self, q, qfpga):
    # ingest images and collect into batches
    # pack each batch as 1 stream
    # dispatch batches to the ingest_workers to copy remote->local buf
    pub = self.get_pub()
    while True:
      try:
        sId = self.get_next_stream_id(self.sub_1, pub)
        inbufIdxs = []
        obj_ids = []

        for i in range(self._bsz):
          obj_id = self.get_id(0, self.sub_0, pub)
          if obj_id is None:
            # send shutdown message
            for p in self._ingestWorkers:
              q.put(None)
            return

          inbufIdx = (sId * self._bsz) + i
          inbufIdxs.append(inbufIdx)
          obj_ids.append(obj_id)

        # push to ingest_worker to copy data from remote buf to local
        q.put((obj_ids, inbufIdxs, sId))
      except Exception as e:
        logger.exception("fpga ingest error")

  def wait(self, q, outputNames):
    # init our own socket
    # (sharing sockets over threads can lead to weirdness)
    pub = self.get_pub()

    numProcessed = 0
    while True:
      try:
        payload = q.get()
        if payload is None:
          break
        requests, sId = payload

        numProcessed += len(requests)
        self._fpgaRT.get_result(sId)

        meta = {
          'id': requests[0]['id'],
          'requests': requests,
          'outputs' : outputNames
        }
        self.put(meta, np.concatenate(self._outputBuffers[sId], axis=None), pub=pub)
      except Exception as e:
        logger.error("FPGA wait error : {}".format(str(e)))

    print("fpga is ending %s" % self._outputs[0])
    self.end(0)

  def run(self):
    firstInputName = next(iterkeys(self._compJson.getInputs()))
    outputNames = list(iterkeys(self._compJson.getOutputs()))

    while True:
      try:
        payload = self._qfpga.get()
        if payload is None:
          break

        (metas, inbufIdxs, sId) = payload
        # print("FPGA get input : {} ".format(metas))
        # print("FPGA get Input : {} {}".format(inbufIdxs, sId))
        # print("FPGA before exec : {} {}".format(len(self._inputBuffers), len(self._outputBuffers)))
        # print("FPGA before exec len of output sid : {}".format(len(self._outputBuffers[sId])))

        input_ptrs = []
        for i, inbufIdx in enumerate(inbufIdxs):
          nparr_view = np.frombuffer(self._inputBuffers[inbufIdx].get_obj(),
            dtype = np.float32)
          input_ptrs.append(nparr_view)

        self._fpgaRT.exec_async( \
          {firstInputName: input_ptrs},
          {outputNames[0]: self._outputBuffers[sId][0],
           outputNames[1]: self._outputBuffers[sId][1]},
           sId)

        self._qwait.put((metas, sId))
      except Exception as e:
        logger.exception("FPGA run error - {}".format(str(e)))

    self.finish()

  def finish(self):
    self._qwait.put(None)
    self._waitThread.join()
    for p in self._ingestWorkers:
      p.terminate()
