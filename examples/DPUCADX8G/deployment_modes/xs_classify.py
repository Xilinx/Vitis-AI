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

import os, signal, time
import logging
import multiprocessing as mp
from threading import Thread
from vai.dpuv1.rt import xdnn_io
from vai.dpuv1.rt import xstream, logging_mp
from vai.dpuv1.rt.xsnodes import fpga, pre, post, grapher
from multiprocessing import Queue

log_file = os.environ['VAI_ALVEO_ROOT'] + "/vai/dpuv1/rt/logging.ini"
logging_mp.setup_logger(log_file, 'xstream')

########################################################
#
########################################################

def request_process(args, img_paths, inchan, outchan):
  q = Queue(100) # only allow this many requests in flight at once

  def throttler():
    xsResult = xstream.Subscribe(outchan)
    while True:
      msg = q.get()
      if msg == None:
        break
      xsResult.get()
  t = Thread(target=throttler)
  t.start()

  # pad num images to be multiple of batch_sz
  if len(img_paths) % args['batch_sz']:
    npad = args['batch_sz'] - (len(img_paths) % args['batch_sz'])
    for i in range(npad):
      img_paths.append(img_paths[-1])

  xs = xstream.Publisher()
  for inum, fpath in enumerate(img_paths):
    (buf, objId) = xs.new(1,
      obj_id="img_%d" % inum,
      meta = {
        'id': inum,
        'path': fpath
      })
    q.put('+')
    xs.put(inchan, objId)

  q.put(None)
  t.join()

########################################################
# "Main"
########################################################

def run(args=None):
  if not args:
    parser = xdnn_io.default_parser_args()
    parser.add_argument('--numprepproc', type=int, default=1,
                        help='# parallel procs to decode/quantize images')
    parser.add_argument('--numstream', type=int, default=6,
                        help='number of FPGA streams')
    parser.add_argument('--deviceID', type=int, default=0,
                        help='FPGA no. -> FPGA ID to use multiple FPGAs')
    parser.add_argument('--benchmarkmode', type=int, default=0,
                        help='bypass pre/post processing for benchmarking')
    parser.add_argument('--startxstream', default=False,
                        action='store_true',
                        help='automatically start obj store server')
    parser.add_argument('--servermode', default=False,
                        action='store_true',
                        help='accept images from another process')
    args = parser.parse_args()
    args = xdnn_io.make_dict_args(args)
    args['preprocseq'] = [
          ('resize', (224, 224)),
          ('meansub', [104.007, 116.669, 122.679]),
          ('chtranspose', (2, 0, 1))]

  # start object store
  # (make sure to 'pip install pyarrow')
  xserver = None
  if args['startxstream']:
    xserver = xstream.Server()

  graph = grapher.Graph("imagenet")
  graph.node("prep", pre.Node, args)
  graph.node("fpga", fpga.Node, args)
  graph.node("post", post.Node, args)

  graph.edge("START", None, "prep")
  graph.edge("prep", "prep", "fpga")
  graph.edge("fpga", "fpga", "post")
  graph.edge("DONE", "post", "fpga")
  graph.edge("DONE", "post", None)

  if not args['servermode']:
    graph.serve(background=True)
    img_paths = xdnn_io.getFilePaths(args['images'])
    reqProc = mp.Process(target=request_process,
      args=(args, img_paths, graph._in[0], graph._out[0],))
    reqProc.start()
    reqProc.join()
    graph.stop(kill=False)
  else:
    print("Serving %s -> %s" % (graph._in[0], graph._out[0]))
    graph.serve()

if __name__ == '__main__':
  run()
