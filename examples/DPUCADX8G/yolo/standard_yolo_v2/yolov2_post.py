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
import time
import timeit
from six import itervalues, next

import numpy as np
import zmq
import os, sys

from vai.dpuv1.rt import xdnn, xdnn_io, xstream
from vai.dpuv1.rt.xsnodes.base import XstreamNode
from vai.dpuv1.utils.postproc import yolo
from yolo_utils import bias_selector, saveDetectionDarknetStyle, yolo_parser_args
from yolo_utils import draw_boxes, generate_colors
from get_mAP_darknet import calc_detector_mAP

import caffe

class ZmqResultPublisher:
  def __init__(self, devid):
    import zmq
    self.context = zmq.Context()
    self.socket = self.context.socket(zmq.PUB)
    self.socket.bind("tcp://*:55{}5".format(devid))

  def send(self, data):
    self.socket.send(data)

class Node(XstreamNode):
  def __init__(self, name='post', inputs=['post'], outputs=['DONE'], args=None):
    super(Node, self).__init__(\
      name=name, inputs=inputs, outputs=outputs, args=args)

  def initialize(self, args):
    self.numProcessed = 0
    self.startTime = timeit.default_timer()
    self.net = caffe.Net(args['deploymodel'], args['caffemodel'], caffe.TEST)
    self.netOut = np.empty((args['batch_sz'],) + self.net.blobs['layer31-conv'].data.shape[1:], dtype=np.float32)
    self.biases = bias_selector(args)
    self._args['net_h'] = self.net.blobs['data'].data.shape[2]
    self._args['net_w'] = self.net.blobs['data'].data.shape[3]
    self.fpgaOutputShapes = list(itervalues(xdnn.CompilerJsonParser(self._args['netcfg']).getOutputs()))
    for i in range(len(self.fpgaOutputShapes)):
      self.fpgaOutputShapes[i][0] = self._args['batch_sz']
    
    # indices for unpacking concatenated arrays to individual array.
    self.buf_indices = [0]
    for i, outputShape in enumerate(self.fpgaOutputShapes):
      self.buf_indices.append(self.buf_indices[-1] + np.prod(outputShape))
    print("Post is starting loop")
    self.run()


  def _run(self, imgList, imgShape, fpgaOutput):
    if self.numProcessed == 0:
      self.startTime = timeit.default_timer()
      self.labels = xdnn_io.get_labels(self._args['labels'])
      self.colors = generate_colors(len(self.labels))
      self.zmqPub = None
      if self._args['zmqpub']:
        self.zmqPub = ZmqResultPublisher(self._args['deviceID'])
      self.goldenMap = None
      if self._args['golden']:
        #self.goldenMap = xdnn_io.getGoldenMap(self._args['golden'])
        self.top5Count = 0
        self.top1Count = 0

    bboxes = yolo.yolov2_postproc([fpgaOutput], self._args, imgShape, biases=self.biases)

    #if self._args['servermode']:
    return bboxes
    #logger.info("POST BLO - " + str(bboxes))

  def run(self):
    super(Node, self).run()

  def loop(self, payload):
    if self.numProcessed == 0:
      self.startTime = timeit.default_timer()

    (meta, buf) = payload
    # print("POST get payload : {} ".format(meta))   
    imgList = []
    imgShape = []

    buf2 = np.frombuffer(buf, dtype=np.float32)
    # print("buf2 details : ", buf2.shape, buf2.dtype)
    # print("buf indices : ", self.buf_indices)
    # print("fpgaOutput shapes : ", self.fpgaOutputShapes)

    bufs = []
    for i in range(len(self.buf_indices[:-1])):
      tmp = buf2[self.buf_indices[i]:self.buf_indices[i+1]].reshape(self.fpgaOutputShapes[i]).copy()
      #print("tmp : ", tmp.shape, tmp.dtype, tmp.flags)
      bufs.append(tmp)

    for ri,r in enumerate(meta['requests']):
      imgList.append(r['path'])
      imgShape.append(r['image_shape'])

    if not self._args["benchmarkmode"]:
      # buf is a list containing multiple blobs
      for b in range(self._args['batch_sz']):
        for idx, bname in enumerate(meta['outputs']): #(layer25-conv, layer27-conv)
            # print("Adding to layer : ", bname, self.net.blobs[bname].data.shape, bufs[idx][b].shape)
            self.net.blobs[bname].data[...] = bufs[idx][b, ...]
        _ = self.net.forward(start='layer28-reorg', end='layer31-conv')
        self.netOut[b, ...] = self.net.blobs['layer31-conv'].data[...]
      
    #  fpgaOutput = np.copy(np.frombuffer(buf, dtype=np.float32)\
    #    .reshape(self.fpgaOutputShape))
      # print("Going to Post run")  
      image_detections = self._run(imgList, imgShape, self.netOut)  # N images with K detections per image, each detection is a dict... list of list of dict
      #for i in range(len(image_detections)):
      #  print("{} boxes detected in image : {}".format(len(image_detections[i]), imgList[i]))

      boxes = image_detections
      if(self._args['results_dir']):
        for i in range(len(imgShape)):
          filename = os.path.splitext(os.path.basename(imgList[i]))[0]
          out_file_txt = os.path.join(self._args['results_dir'], filename + '.txt')
          # print("Saving {} boxes to {}".format(len(boxes[i]), out_file_txt)); sys.stdout.flush()
          saveDetectionDarknetStyle(out_file_txt, boxes[i], imgShape[i])
          if(self._args['visualize']):
            out_file_png = os.path.join(self._args['results_dir'], filename + '.png')
            # print("Saving result to {}".format(out_file_png)); sys.stdout.flush()
            draw_boxes(imgList[i], boxes[i], self.labels, self.colors, out_file_png)

      #[[{"classid": 21, "ll": {"y": 663, "x": 333}, "ur": {"y": 238, "x": 991}, "prob": 0.6764760613441467, "label": "bear"}]]

      for ri,r in enumerate(meta['requests']):
        detections = image_detections[ri] # Examine result for first image
        boxes = []
        # detection will be a dict
        for detection in detections:
          x1 = detection["ll"]["x"]
          #y1 = detection["ll"]["y"] # ONEHACK to conform to the way facedetect does corners
          y1 = detection["ur"]["y"]
          x2 = detection["ur"]["x"]
          #y2 = detection["ur"]["y"]
          y2 = detection["ll"]["y"]
          label = detection["classid"]
          boxes.append([x1,y1,x2,y2,label])

        meta['requests'][ri]['boxes'] = boxes
        meta['requests'][ri]['callback'] = self._callback

    self.numProcessed += len(meta['requests'])

    # TODO shouldn't meta always have requests?
    if 'requests' in meta:
      for r in meta['requests']:
        self.put(r)
    del buf
    del payload

  def finish(self):
    t  = timeit.default_timer()
    self._args['timerQ'].put(t)
    print("yolopost is ending %s" % self._outputs[0])
    self.end(0)
