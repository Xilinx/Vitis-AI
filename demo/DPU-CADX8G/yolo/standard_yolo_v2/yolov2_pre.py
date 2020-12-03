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
import json
import traceback
import multiprocessing as mp
import multiprocessing.pool
import numpy as np
from vai.dpuv1.rt import xdnn, xdnn_io
from vai.dpuv1.rt import xstream
from vai.dpuv1.rt.xsnodes.base import XstreamNode
import base64
import cv2
import logging
from six import itervalues, next

logger = logging.getLogger(__name__)

# configure no-daemon mode so we can create a child process pool
class NoDaemonProcess(mp.Process):
  # make 'daemon' attribute always return False
  def _get_daemon(self):
      return False
  def _set_daemon(self, value):
      pass
  daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NoDaemonProcessPool(multiprocessing.pool.Pool):
  Process = NoDaemonProcess

g_preInst = None
def init_pre_process(name, inputs, outputs, args):
  global g_preInst
  g_preInst = Node(name, inputs, outputs, args, master=False)
def run_pre_process(img):
  global g_preInst
  return g_preInst._run(img)

class Node(XstreamNode):
  def __init__(self, name='prep', inputs=['START'], outputs=['fpga'], args=None, master=True):
    self.master = master
    super(Node, self).__init__(\
      name=name, inputs=inputs, outputs=outputs, args=args)

  def initialize(self, args):
    self._xsIn = xstream.Base()
    self._compJson = xdnn.CompilerJsonParser(args['netcfg'])
    self._firstInputShape = next(itervalues(self._compJson.getInputs()))

    # For now don't use a mean array, in theory this should avoid broadcasting, but harder to pass in from recipe
    # Can be hacked in, if that performance boost is necessary
    #HWC format as this is the native format that comes out of jpeg decode
    #self._meanarr = np.zeros ( (self._firstInputShape[2], self._firstInputShape[3], self._firstInputShape[1],), dtype = np.float32, order='C' )
    #self._meanarr += args['img_mean']

    if self.master:
      print("Pre is starting loop")
      self.run()

  def _run(self, objId):
    (meta, inbuf) = self._xsIn.obj_get(objId)
    # print("PRE get input : {} ".format(meta))

    try:
      if 'path' in meta:
        img = str(meta['path'])
      else:
        img = np.frombuffer(inbuf, getattr(np, meta['dtype']))
        img = img.reshape(
          int(meta['image_height']),
          int(meta['image_width']),
          int(meta['image_channels'])
        )
        meta['path'] = 'dummy_path' # TODO remove the dependency on this key

        # put encoded image in meta for return to client
        thumb = xdnn_io.makeThumbnail(img,
          max(self._firstInputShape[1], self._firstInputShape[2]))
        meta['resized_shape'] = thumb.shape
        retval, img_str = cv2.imencode(".jpg", thumb)
        if retval:
          base64_str = base64.b64encode(img_str)
          meta['img'] = base64_str.decode('utf-8')

      np_arr = np.zeros(tuple(self._firstInputShape[1:]),
        dtype=np.float32, order='C')

      if not self._args['benchmarkmode']:
        # print("PRE loading img : {}".format(img))

        np_arr[:], meta['image_shape'] = xdnn_io.loadYoloImageBlobFromFile(img, 
            int(self._args['net_h']), int(self._args['net_w']))


      # np_arr has the shape (C,H,W)
      # print("PRE put input : {} ".format(meta))
      self.put(meta, np_arr)
    except Exception as e:
      print("ERROR : {}".format(str(e)))
      logger.exception("pre error %s - %s" % (meta['id']), str(e))


  def run(self):
    # this loop dispatches work to worker processes
    p = NoDaemonProcessPool(initializer = init_pre_process,
      initargs = (self._name, self._inputs, self._outputs, self._args),
      processes = self._args['numprepproc'])

    while True:
      try:
        obj_id = self.get_id()
        if obj_id is None:
          break

        # dispatch job to one of the workers
        p.map_async(run_pre_process, [obj_id])
      except Exception as e:
        logger.exception(e)

    p.close()
    p.join()

    print("pre is ending %s" % self._outputs[0])
    self.end(0)
