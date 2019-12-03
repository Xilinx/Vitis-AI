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
import math

from vai.dpuv1.rt import xstream, xdnn_io, xdnn
from vai.dpuv1.rt.vitis.python.dpu.runner import Runner
from vai.dpuv1.utils.postproc import yolo
from yolo_utils import bias_selector, saveDetectionDarknetStyle, yolo_parser_args
from yolo_utils import draw_boxes, generate_colors
from get_mAP_darknet import calc_detector_mAP

sys.path.insert(0, os.environ["VAI_ALVEO_ROOT"] + '/examples/deployment_modes')
from mp_classify_vitis import *

logging.basicConfig(filename='xs_detect_vitis.log', filemode='w', level=logging.DEBUG)

class yoloDispatcher(Dispatcher):

  @staticmethod
  def _run(work):
    try:
      (idx, images, args) = work
      chanIdx = idx % Dispatcher.nWorkers
      token = pid2TokenStr()
      shape = Dispatcher.inshape

      image_shapes = []
      for i, img in enumerate(images):
        cached = Dispatcher.inBlobCache.get(img)
        if cached is None:
          Dispatcher.inBlob[token][i, ...], img_shape = xdnn_io.loadYoloImageBlobFromFile(img, shape[2], shape[3])
          Dispatcher.inBlobCache.set(img, (Dispatcher.inBlob[token][i].copy(), img_shape))
          image_shapes.append(img_shape)
        else:
          Dispatcher.inBlob[token][i, ...] = cached[0]
          image_shapes.append(cached[1])

      meta = { 'id': idx, 'from': token, 'shape': shape, 'images': images, 'image_shapes': image_shapes }
      if idx % 1000 == 0:
        print("Put query %d to objstore" % idx)
        sys.stdout.flush()

      Dispatcher.xspub[token].put_blob(chanIdx2Str(chanIdx), Dispatcher.inBlob[token], meta)
      Dispatcher.xstoken[token].get_msg()
    except Exception as e:
      logging.error("Producer exception " + str(e))

  def run(self, work):
    self.pool.map_async(yoloDispatcher._run, work)

class yoloWorkerPool(WorkerPool):
  def __init__(self, rundir, nWorkers, workerArgs):
    self.xspub = xstream.Publisher()
    self.workers = []
    self.wq = mp.Queue()
    for wi in range(nWorkers):
      w = mp.Process(target=yoloWorkerPool.run, args=(rundir, wi, self.wq, workerArgs,))
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
    labels = xdnn_io.get_labels(args['labels'])
    if   args['yolo_version'] == 'v2': yolo_postproc = yolo.yolov2_postproc
    elif args['yolo_version'] == 'v3': yolo_postproc = yolo.yolov3_postproc
    else: assert args['yolo_version'] in ('v2', 'v3'), "--yolo_version should be <v2|v3>"

    biases = bias_selector(args)
    if(args['visualize']): colors = generate_colors(len(labels))

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

        boxes = yolo_postproc(fpgaBlobs[1], args, meta['image_shapes'], biases=biases)

        if(not args['profile']):
          for i in range(min(batchsz, len(meta['image_shapes']))):
            print("Detected {} boxes in {}".format(len(boxes[i]), meta['images'][i]), flush=True)

        # Save the result
        if(args['results_dir']):
          for i in range(min(batchsz, len(meta['image_shapes']))):
            fname = meta['images'][i]
            filename = os.path.splitext(os.path.basename(fname))[0]
            out_file_txt = os.path.join(args['results_dir'], filename + '.txt')
            print("Saving {} boxes to {}".format(len(boxes[i]), out_file_txt)); sys.stdout.flush()
            saveDetectionDarknetStyle(out_file_txt, boxes[i], meta['image_shapes'][i])

            if(args['visualize']):
              out_file_png = os.path.join(args['results_dir'], filename + '.png')
              print("Saving result to {}".format(out_file_png)); sys.stdout.flush()
              draw_boxes(fname, boxes[i], labels, colors, out_file_png)

        if meta['id'] % 1000 == 0:
          print("Recvd query %d" % meta['id'])
          sys.stdout.flush()

        del data
        del buf
        del payload

        xspub.send(meta['from'], "success")

      except Exception as e:
        logging.error("Worker exception " + str(e))

def main():
  parser = xdnn_io.default_parser_args()
  parser = yolo_parser_args(parser)
  args = parser.parse_args()
  args = xdnn_io.make_dict_args(args)

  g_nDispatchers = args['numprepproc']
  g_nWorkers = args['numworkers']

  # Setup the environment
  images = xdnn_io.getFilePaths(args['images'])
  if(args['golden'] or args['visualize']):
    assert args['labels'], "Provide --labels to compute mAP."
    assert args['results_dir'], "For accuracy measurements, provide --results_dir to save the detections."

  # start comms
  xserver = xstream.Server()

  # acquire resources
  fmaster = FpgaMaster(args['vitis_rundir'])

  # update batch size
  inshape = list(fmaster.inshape)
  if args['batch_sz'] != -1:
    inshape[0] = args['batch_sz']

  args['net_h'] = inshape[2]
  args['net_w'] = inshape[3]

  # spawn dispatchers
  dispatcher = yoloDispatcher(g_nDispatchers, g_nWorkers, inshape)

  # spawn workers
  workers = yoloWorkerPool(args['vitis_rundir']+"_worker", g_nWorkers, args)

  # send work to system
  g_nQueries = int(np.ceil(len(images) / inshape[0]))
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

  # cleanup
  del workers
  del fmaster
  del xserver

  # mAP calculation
  if(args['golden']):
    print()
    print("Computing mAP score  : ")
    labels = xdnn_io.get_labels(args['labels'])
    print("Class names are  : {} ".format(labels))
    mAP = calc_detector_mAP(args['results_dir'], args['golden'], len(labels), labels, args['prob_threshold'], args['iouthresh'])
    sys.stdout.flush()

if __name__ == '__main__':
  main()
