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
from six import itervalues,iterkeys

import sys, os
import timeit
import numpy as np
import multiprocessing as mp
import ctypes
import signal
import threading
import time

from vai.dpuv1.rt import xdnn, xdnn_io
from vai.dpuv1.rt.vitis.python.dpu.runner import Runner

###################################################
# Pre-process
###################################################
class UserPreProcess():
  def __init__(self, args, img_paths,  input_shapes, shared_trans_arrs):
    np.random.seed(123)  # for reproducibility
    self._args = args
    self._firstInputShape = input_shapes[0]
    self._shared_trans_arrs = shared_trans_arrs

    self._imgpaths = img_paths
    current = mp.current_process()
    self._procid = (int(current._identity[0]) - 1) % args['numprepproc']

    #HWC format as this is the native format that comes out of jpeg decode
    self._meanarr = np.zeros ( (self._firstInputShape[2], self._firstInputShape[3], self._firstInputShape[1],), dtype = np.float32, order='C' )
    self._meanarr += args['img_mean']

  def run(self, inum_chunk):

    write_slot = self._shared_trans_arrs.openWriteId()
    write_arrs = self._shared_trans_arrs.accessNumpyBuffer(write_slot)

    if not self._args['benchmarkmode']:
      for i, inum in enumerate(inum_chunk):
        write_arrs[0][i][:], shape = xdnn_io.loadImageBlobFromFile(self._imgpaths[inum], self._args['img_raw_scale'], self._meanarr,
                                               self._args['img_input_scale'], self._firstInputShape[2], self._firstInputShape[3])
        write_arrs[-1][i][0] = inum
        write_arrs[-1][i][1:4] = shape

      # Fill -1 for unfilled image slots in whole batch
      write_arrs[-1][len(inum_chunk):][:] = -1

    self._shared_trans_arrs.closeWriteId(write_slot)

###################################################
# Post-process
###################################################

class ZmqResultPublisher:
  def __init__(self, devid):
    import zmq
    self.context = zmq.Context()
    self.socket = self.context.socket(zmq.PUB)
    self.socket.bind("tcp://*:55{}5".format(devid))

  def send(self, data):
    self.socket.send(data)

class UserPostProcess():
  def __init__(self,  args, img_paths,  fpgaOutputs, output_shapes,shared_output_arrs):
    self.args = args
    self.img_paths = img_paths

    self.fpgaOutputs = fpgaOutputs
    self.output_shapes = output_shapes
    self._shared_output_arrs = shared_output_arrs

    self.numProcessed = 0
    self.startTime = timeit.default_timer()
    self.cpuOp = xdnn.XDNNCPUOp(self.args['weights']);

  #
  # This function post-processes FPGA output:
  # 1) Compute the final FC + Softmax layers
  # 2) Print classification & accuracy
  #
  def run(self, imgList, fpgaOutput_list, fpgaOutputShape_list, shape_list):
    fpgaOutput  = fpgaOutput_list[0]
    fpgaOutputShape  = fpgaOutputShape_list[0]
    if self.numProcessed == 0:
      self.startTime = timeit.default_timer()
      self.labels = xdnn_io.get_labels(self.args['labels'])
      self.zmqPub = None
      if self.args['zmqpub']:
        self.zmqPub = ZmqResultPublisher(self.args['deviceID'])
      self.goldenMap = None
      if self.args['golden']:
        self.goldenMap = xdnn_io.getGoldenMap(self.args['golden'])
        self.top5Count = 0
        self.top1Count = 0
      self.fcOutput = np.empty((self.args['batch_sz'], self.args['outsz'],),
        dtype=np.float32, order='C')

    self.numProcessed += len(imgList)

    npout_view = fpgaOutput
    self.cpuOp.computeFC(npout_view, self.fcOutput)
    smaxOutput = self.cpuOp.computeSoftmax(self.fcOutput)

    if self.args['golden']:
      for i,p in enumerate ( imgList ):
        #topk = xdnn_io.getTopK( smaxOutput[i], self.labels, 1)
        #print(imgList[i], topk)
        self.top1Count += xdnn_io.isTopK(\
          smaxOutput[i], self.goldenMap, p, self.labels, 1)
        self.top5Count += xdnn_io.isTopK(\
          smaxOutput[i], self.goldenMap, p, self.labels, 5)

    if self.zmqPub is not None:
      predictMsg = xdnn_io.getClassification(\
        smaxOutput, imgList, self.labels, zmqPub=True)
      self.zmqPub.send(predictMsg)

  def loop(self):
    fpgaOutputShape = self.output_shapes[0]
    fpgaOutputShape[0] = self.args['batch_sz']
    frame_count = 0

    while True:
      read_slot = self._shared_output_arrs.openReadId()

      if read_slot is None:
          break

      read_slot_arrs = self._shared_output_arrs.accessNumpyBuffer(read_slot)
      # array_sum = [read_slot_arrs[0][i].sum() for i in range(self.args['batch_sz'])]

      imgList = []
      shape_list = []
      num_images = (read_slot_arrs[-1].shape)[0]
      for image_num in range(num_images):
          image_id = read_slot_arrs[-1][image_num][0]

          if image_id == -1:
              break

          imgList.append(self.img_paths[int(image_id)])
          shape_list.append(read_slot_arrs[-1][image_num][1:4])


      if self.args["benchmarkmode"]:
        self.numProcessed += len(imgList)
        self._shared_output_arrs.closeReadId(read_slot)
        continue

      self.run(imgList,read_slot_arrs[0:-1], [fpgaOutputShape], shape_list)
      self._shared_output_arrs.closeReadId(read_slot)

    self.finish()

  def finish(self):
    print( "%g images/s" % ( float(self.numProcessed) / (timeit.default_timer() - self.startTime )  ))
    if self.args['golden'] and self.numProcessed:
      print("\nAverage accuracy (n=%d) Top-1: %.1f%%, Top-5: %.1f%%\n" \
        % (self.numProcessed,
           float(self.top1Count)/float(self.numProcessed)*100.,
           float(self.top5Count)/float(self.numProcessed)*100.))

###################################################
# Instantiate pre/post processes,
# allow user to register own classes
###################################################

g_preClass = UserPreProcess
g_postClass = UserPostProcess
g_preInst = None
g_postInst = None

def register_pre(preClass):
  global g_preClass
  g_preClass = preClass

def register_post(postClass):
  global g_postClass
  g_postClass = postClass


def init_pre_process(args, img_paths,  input_shapes, shared_trans_arrs):
  global g_preClass
  global g_preInst
  g_preInst = g_preClass(args,   img_paths,  input_shapes, shared_trans_arrs)

def run_pre_process(imgpath_idx):
  global g_preInst
  return g_preInst.run(imgpath_idx)

def post_process( args, img_paths, fpgaOutputs, output_shapes,shared_output_arrs):
  global g_postClass
  global g_postInst
  g_postInst = g_postClass( args, img_paths, fpgaOutputs, output_shapes, shared_output_arrs)
  g_postInst.loop()

###################################################
# FPGA
###################################################

def fpga_wait( runner, q, shared_output_arrs, shared_trans_arrs):
  numProcessed = 0
  while True:
    write_slot, read_slot, jid = q.get()

    if write_slot is None:
      break

    runner.wait(jid)

    #qFpga.put(img_num)
    shared_trans_arrs.closeReadId(read_slot)
    shared_output_arrs.closeWriteId(write_slot)

  shared_output_arrs.close()


def fpga_process(args, num_img,  compJson, shared_trans_arrs,shared_output_arrs):
    runner = Runner(args['vitis_rundir'])
    qWait = mp.Queue(maxsize=100)

    t = threading.Thread(target=fpga_wait, args=(runner, qWait, shared_output_arrs, shared_trans_arrs))
    t.start()

    numProcessed = 0
    startTime = time.time()
    while numProcessed < num_img or args['perpetual']:
        # Get the buffer for fpga output
        write_slot = shared_output_arrs.openWriteId()
        write_slot_arrs = shared_output_arrs.accessNumpyBuffer(write_slot)

        # Get the input buffer for fpga exec
        read_slot = shared_trans_arrs.openReadId()
        if read_slot is None: break
        read_slot_arrs = shared_trans_arrs.accessNumpyBuffer(read_slot)

        # Copy meta data from input to output
        write_slot_arrs[-1][:] = read_slot_arrs[-1][:]

        # Start execution
        jid = runner.execute_async(read_slot_arrs[:-1], write_slot_arrs[:-1])
        # runner.wait(jid)
        qWait.put((write_slot, read_slot, jid))
        #shared_trans_arrs.closeReadId(read_slot)

        numProcessed += 1
        if(args['perpetual'] == False):
            if numProcessed == num_img:
                break


    qWait.put((None, None, None))
    t.join()
    elapsedTime = ( time.time() - startTime )
    print( "FPGA_process: ", float(numProcessed)/elapsedTime, "img/s")

# Current version does copies...
# Assumes all types are np.float32/ctypes.c_float
class SharedMemoryQueue:
    def __init__(self, name, length, buf_shapes_list):

        print("Creating SharedMemoryQueue",name)
        self._name = name
        self._len = length

        # Hard coded for floats...
        self._mem_type = ctypes.c_float
        self._np_type = np.float32

        # put() function copies into the free list
        self._freeList = mp.Queue(length)

        # get() function gets id of open slot. consumer needs to confirm when data is read
        self._readList = mp.Queue(length)

        self._buf_shapes_list = buf_shapes_list
        self._buf_sizes_list = list(map(lambda x: int(np.prod(x)), buf_shapes_list))

        print("Creating Shared Memory with buf_shape_list =", self._buf_shapes_list)

        self._shared_memory_arrs = list()
        for i in range(length):
            buf_list = list()
            for buf_size in self._buf_sizes_list:
                buf_list.append(mp.Array(self._mem_type, buf_size))
            self._shared_memory_arrs.append(buf_list)
            self._freeList.put(i)


    def close(self):
        self._readList.put(None)


    def accessBuffer(self, slot_id):
        return self._shared_memory_arrs[slot_id]


    def accessNumpyBuffer(self, slot_id):
        buf_list = list()
        for i in range(len(self._buf_shapes_list)):
            np_arr = np.frombuffer(self._shared_memory_arrs[slot_id][i].get_obj(), dtype = self._np_type)
            np_arr = np.reshape(np_arr, self._buf_shapes_list[i], order = 'C')
            buf_list.append(np_arr)

        return buf_list


    def openWriteId(self):
        id = self._freeList.get()
        return id


    def closeWriteId(self, id):
        # finished writing slot id
        self._readList.put(id)


    def openReadId(self):
        id = self._readList.get()
        return id


    def closeReadId(self, id):
        # finished reading slot id
        self._freeList.put(id)


    def dump(self):
        for i in range(self._len):
          buf_list = self.accessNumpyBUffer(i)
          for np_arr in buf_list:
              print("Slot=",i,"Array=",j,"Val=",np_arr)


###################################################
# "Main"
###################################################

def run(args=None):
  if not args:
    parser = xdnn_io.default_parser_args()
    parser.add_argument('--numprepproc', type=int, default=1,
                        help='number of parallel processes used to decode and quantize images')
    parser.add_argument('--numstream', type=int, default=16,
                        help='number of FPGA streams')
    parser.add_argument('--deviceID', type=int, default=0,
                        help='FPGA no. -> FPGA ID to run in case multiple FPGAs')
    parser.add_argument('--benchmarkmode', type=int, default=0,
                        help='bypass pre/post processing for benchmarking')
    parser.add_argument('--profile', action='store_true',
                        help='Print average latencies for preproc/exec/postproc')

    args = parser.parse_args()
    args = xdnn_io.make_dict_args(args)

  sharedInputArrs = []
  fpgaOutputs = []

  compilerJSONObj = xdnn.CompilerJsonParser(args['netcfg'])

  input_shapes = [v for k,v in compilerJSONObj.getInputs().items()]
  output_shapes = [v for k,v in compilerJSONObj.getOutputs().items()]

  for in_idx in range(len(input_shapes)):
      input_shapes[in_idx][0] = args['batch_sz']
  for out_idx in range(len(output_shapes)):
      output_shapes[out_idx][0] = args['batch_sz']

  input_sizes = map(lambda x: np.prod(x), input_shapes)
  output_sizes = map(lambda x: np.prod(x), output_shapes)

  num_shared_slots = args['numstream']

  # shared memory from preprocessing to fpga forward
  shared_trans_arrs = SharedMemoryQueue("trans",num_shared_slots*(args['numprepproc']*args['batch_sz']),
          input_shapes +[(args['batch_sz'], 4)])

  # shared memory from fpga forward to postprocessing
  shared_output_arrs = SharedMemoryQueue("output",num_shared_slots, output_shapes + [(args['batch_sz'], 4)])

  # Form list of images to chunks of batch_sz
  img_paths = xdnn_io.getFilePaths(args['images'])
  imgids = list(range(len(img_paths)))
  imgid_chunks = [ imgids[i:i+args['batch_sz']] for i in range(0, len(img_paths), args['batch_sz']) ]

  # Start all processes
  p = mp.Pool(initializer = init_pre_process,
    initargs = (args,  img_paths, input_shapes, shared_trans_arrs, ), processes = args['numprepproc'])

  xdnnProc = mp.Process(target=fpga_process, args=(args, len(imgid_chunks), compilerJSONObj,shared_trans_arrs,shared_output_arrs,))

  postProc = mp.Process(target=post_process, args=(args, img_paths, fpgaOutputs,output_shapes,shared_output_arrs,))
  xdnnProc.start()
  postProc.start()

  t1 = timeit.default_timer()
  if args['perpetual']:
    while True:
      res = [p.map_async(run_pre_process, imgid_chunks)]
      for j in res:
        j.wait()
        del j
  else:
    p.map_async(run_pre_process, imgid_chunks)

  xdnnProc.join()
  postProc.join()

  p.close()
  p.join()
  t2 = timeit.default_timer()
  total_t = t2 - t1
  if(args['profile']):
    print("Total time taken: {} s\n Total images: {}\nAverage FPS: {}".format(total_t, \
            len(img_paths), len(img_paths)/total_t))

if __name__ == '__main__':
  run()
