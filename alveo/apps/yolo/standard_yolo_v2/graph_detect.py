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

# Usage : 
# source <MLsuite>/overlaybins/setup.sh
# python graph_detect.py --xclbin /proj/sdxapps/users/abidk/repos/MLsuite/overlaybins/xdnnv3 --netcfg work.std_yolo_v2/compiler.json --weights work/weights.h5 --quantizecfg work/quantizer.json --images ../../tmp/val100/

import os, signal, time, timeit, sys
import logging
import multiprocessing as mp
from threading import Thread
from vai.dpuv1.rt import xdnn, xdnn_io
from vai.dpuv1.rt import xstream, logging_mp
from vai.dpuv1.rt.xsnodes import fpga, pre, post, grapher
import yolov2_fpga, yolov2_post, yolov2_pre
from multiprocessing import Queue
from six import itervalues, next

from yolo_utils import bias_selector, saveDetectionDarknetStyle, yolo_parser_args
from yolo_utils import draw_boxes, generate_colors
from get_mAP_darknet import calc_detector_mAP

log_file = os.environ['VAI_ALVEO_ROOT'] + "/vai/dpuv1/rt/logging.ini"
logging_mp.setup_logger(log_file, 'xstream')

########################################################
#
########################################################

def request_process(args, img_paths, inchan, outchan):
  # print("Starting REQ PROC")
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
    # print("REQ put data : {} ".format(fpath))

  q.put(None)
  t.join()

########################################################
# "Main"
########################################################

def run(args=None):
  if not args:
    parser = xdnn_io.default_parser_args()
    parser = yolo_parser_args(parser)
    parser.add_argument('--startxstream', default=True,
                        action='store_true',
                        help='automatically start obj store server')
    parser.add_argument('--servermode', default=False,
                        action='store_true',
                        help='accept images from another process')
    parser.add_argument("--deploymodel",  type=str, default='',
                        help='Original prototxt')
    parser.add_argument("--caffemodel",  type=str, default='',
                        help='Original caffemodel')

    args = parser.parse_args()
    args = xdnn_io.make_dict_args(args)
    args['preprocseq'] = [
          ('resize', (224, 224)),
          ('meansub', [104.007, 116.669, 122.679]),
          ('chtranspose', (2, 0, 1))]

  if(args['golden'] or args['visualize']):
    assert args['labels'], "Provide --labels to compute mAP."
    assert args['results_dir'], "For accuracy measurements, provide --results_dir to save the detections."
    labels = xdnn_io.get_labels(args['labels'])
    colors = generate_colors(len(labels))

  args['startxstream']    = True
  args['servermode']      = False

  timerQ = Queue()
  args['timerQ']           = timerQ
  
  compJson = xdnn.CompilerJsonParser(args['netcfg'])
  firstInputShape = next(itervalues(compJson.getInputs()))
  args['net_h'] = firstInputShape[2]
  args['net_w'] = firstInputShape[3]

    # start object store
  # (make sure to 'pip install pyarrow')
  xserver = None
  if args['startxstream']:
    xserver = xstream.Server()

  graph = grapher.Graph("yolo_v2")
  graph.node("prep", yolov2_pre.Node, args)
  graph.node("fpga", yolov2_fpga.Node, args)
  graph.node("post", yolov2_post.Node, args)

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

    t = timeit.default_timer()
    reqProc.start()
    reqProc.join()
    graph.stop(kill=False)
    t2 = args['timerQ'].get()
    full_time = t2 -  t

    args['timerQ'].close()

    print("Total time : {}s for {} images".format(full_time, len(img_paths)))
    print("Average FPS : {} imgs/sec".format(len(img_paths)/full_time))
  else:
    print("Serving %s -> %s" % (graph._in[0], graph._out[0]))
    graph.serve()

  # mAP calculation
  if(args['golden']):
    print(flush=True)
    print("Computing mAP score  : ", flush=True)
    print("Class names are  : {} ".format(labels), flush=True)
    mAP = calc_detector_mAP(args['results_dir'], args['golden'], len(labels), labels,\
            args['prob_threshold'], args['mapiouthresh'], args['points'])
    sys.stdout.flush()


if __name__ == '__main__':
  run()
