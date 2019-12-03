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
import os, sys
import timeit
import numpy as np
import multiprocessing as mp
import ctypes
import threading
import time
import logging as log
from yolo_utils import darknet_style_xywh, cornersToxywh,sigmoid,softmax,generate_colors,draw_boxes
from yolo_utils import bias_selector, saveDetectionDarknetStyle, yolo_parser_args
from yolo_utils import draw_boxes, generate_colors
from get_mAP_darknet import calc_detector_mAP

from vai.dpuv1.utils.postproc import yolo
from vai.dpuv1.rt import xdnn, xdnn_io

sys.path.insert(0, os.environ["VAI_ALVEO_ROOT"] + '/examples/deployment_modes')
import mp_classify as mp_classify
sys.path.insert(0, os.environ["VAI_ALVEO_ROOT"] + '/apps/yolo')

class YoloPreProcess(mp_classify.UserPreProcess):
  def run(self, inum_chunk):
    write_slot = self._shared_trans_arrs.openWriteId()
    write_arrs = self._shared_trans_arrs.accessNumpyBuffer(write_slot)

    if not self._args['benchmarkmode']:
      for i, inum in enumerate(inum_chunk):
        write_arrs[0][i][:], shape = xdnn_io.loadYoloImageBlobFromFile(self._imgpaths[inum], self._firstInputShape[2], self._firstInputShape[3])
        write_arrs[-1][i][0] = inum
        write_arrs[-1][i][1:4] = shape

      # Fill -1 for unfilled image slots in whole batch
      write_arrs[-1][len(inum_chunk):][:] = -1

    self._shared_trans_arrs.closeWriteId(write_slot)

class YoloPostProcess(mp_classify.UserPostProcess):
  def loop(self):
    fpgaOutputShapes = []
    for idx in range(len( self.output_shapes)):
        fpgaOutputShape_l = self.output_shapes[idx]
        fpgaOutputShape_l[0] = self.args['batch_sz']
        fpgaOutputShapes.append(fpgaOutputShape_l)

    if   self.args['yolo_version'] == 'v2': self.yolo_postproc = yolo.yolov2_postproc
    elif self.args['yolo_version'] == 'v3': self.yolo_postproc = yolo.yolov3_postproc

    self.biases = bias_selector(self.args)
    self.labels = xdnn_io.get_labels(self.args['labels'])
    self.colors = generate_colors(len(self.labels))

    while True:
      read_slot = self._shared_output_arrs.openReadId()
      if read_slot is None:
          break

      read_slot_arrs = self._shared_output_arrs.accessNumpyBuffer(read_slot)
      imgList = []
      shape_list = []
      #image_id = self._qFrom.get()
      num_images = (read_slot_arrs[-1].shape)[0]
      for image_num in range(num_images):
          image_id = read_slot_arrs[-1][image_num][0]

          if image_id == -1:
              break
          imgList.append(self.img_paths[int(image_id)])
          shape_list.append(read_slot_arrs[-1][image_num][1:4])

      if self.args["benchmarkmode"]:
        self.numProcessed += len(imgList)
        #self.streamQ.put(sId)
        self._shared_output_arrs.closeReadId(read_slot)
        continue

      self.run(imgList,read_slot_arrs[0:-1], fpgaOutputShapes, shape_list)
      self._shared_output_arrs.closeReadId(read_slot)

    self.finish()

  def run(self, imgList, fpgaOutput_list, fpgaOutputShape_list, shapeArr):
    if self.numProcessed == 0:
      self.zmqPub = None
      if self.args['zmqpub']:
        self.zmqPub = mp_classify.ZmqResultPublisher(self.args['deviceID'])
      self.goldenMap = None


    self.numProcessed += len(imgList)
    bboxlist_for_images = self.yolo_postproc(fpgaOutput_list, args, shapeArr, biases=self.biases)

    if(not self.args['profile']):
      for i in range(min(self.args['batch_sz'], len(shapeArr))):
        print("Detected {} boxes in {}".format(len(bboxlist_for_images[i]), imgList[i]))


    if(self.args['results_dir']):
      boxes = bboxlist_for_images
      for i in range(min(self.args['batch_sz'], len(shapeArr))):
        filename = os.path.splitext(os.path.basename(imgList[i]))[0]
        out_file_txt = os.path.join(self.args['results_dir'], filename + '.txt')
        print("Saving {} boxes to {}".format(len(boxes[i]), out_file_txt)); sys.stdout.flush()
        saveDetectionDarknetStyle(out_file_txt, boxes[i], shapeArr[i])

        if(self.args['visualize']):
          out_file_png = os.path.join(self.args['results_dir'], filename + '.png')
          print("Saving result to {}".format(out_file_png)); sys.stdout.flush()
          draw_boxes(imgList[i], boxes[i], self.labels, self.colors, out_file_png)

  def finish(self):
    print("[XDNN] Total Images Processed : {}".format(self.numProcessed)); sys.stdout.flush()

    # mAP calculation
    if(args['golden']):
      print()
      print("Computing mAP score  : ")
      print("Class names are  : {} ".format(self.labels))
      mAP = calc_detector_mAP(args['results_dir'], args['golden'], len(self.labels), self.labels, self.args['prob_threshold'], self.args['iouthresh'])
      sys.stdout.flush()

mp_classify.register_pre(YoloPreProcess)
mp_classify.register_post(YoloPostProcess)

if __name__ == '__main__':
  parser = xdnn_io.default_parser_args()
  parser = yolo_parser_args(parser)
  args = parser.parse_args()
  args = xdnn_io.make_dict_args(args)

  if(args['golden'] or args['visualize']):
    assert args['labels'], "Provide --labels to compute mAP."
    assert args['results_dir'], "For accuracy measurements, provide --results_dir to save the detections."

  compilerJSONObj = xdnn.CompilerJsonParser(args['netcfg'])

  input_shapes = [v for k,v in compilerJSONObj.getInputs().items()]
  output_shapes = [v for k,v in compilerJSONObj.getOutputs().items()]

  for out_idx in range(len(output_shapes)):
      output_shapes[out_idx][0] = args['batch_sz']

  input_sizes  = map(lambda x: np.prod(x), input_shapes)
  output_sizes = map(lambda x: np.prod(x), output_shapes)

  out_w = output_shapes[0][2]
  out_h = output_shapes[0][3]

  args['net_w'] = int(input_shapes[0][2])
  args['net_h'] = int(input_shapes[0][3])
  args['out_w'] = int(out_w)
  args['out_h'] = int(out_h)
  args['coords'] = 4
  args['beginoffset'] = (args['coords']+1) * int(out_w * out_h)
  args['groups'] = int(out_w * out_h)
  args['batchstride'] = args['groups']*(args['outsz']+args['coords']+1)
  args['groupstride'] = 1
  args['classes'] = args['outsz']
  args['bboxplanes'] = args['anchorCnt']

  print ("running yolo_model : ", args['yolo_model'])

  mp_classify.run(args)
