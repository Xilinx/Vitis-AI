#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2019, Xilinx, Inc.
#

from __future__ import print_function 
import os, sys
from six import itervalues, iteritems
from ctypes import *
import numpy as np
import timeit
import waa_rt
import cv2
sys.path.append("../../yolo")
from vai.dpuv1.rt import xdnn, xdnn_io
from vai.dpuv1.rt.vitis.python.dpu.runner import Runner
from vai.dpuv1.utils.postproc import yolo
from yolo_utils import bias_selector, saveDetectionDarknetStyle, yolo_parser_args
from yolo_utils import draw_boxes, generate_colors
from get_mAP_darknet import calc_detector_mAP
import multiprocessing as mp

def pre_process(q_img, q_shape,args):

  xclbin_p=str(args['xclbin']+"/xdnn_v3_96x16_2pe_8b_9mb_bank03.xclbin")
  kernelName_p="pp_pipeline_accel"
  deviceIdx_p=args['deviceid']
  handle_p = waa_rt.PreProcess(xclbin_p,kernelName_p,deviceIdx_p, 1)
  if handle_p == -1:
    print("Unable to Create handle for pre-processing kernel. Only U200 device is supported")
    sys.exit()
  batch_sz = args['batch_sz']
  img_paths = xdnn_io.getFilePaths(args['images'])
  print("Pre-processing handle created. Populating Queue")
  for i in range(0, len(img_paths), batch_sz):
    for j, p in enumerate(img_paths[i:i + batch_sz]):
      arr, shape = handle_p.preprocess_input(p)
      q_img.put(arr)
      q_shape.put(shape)

def process_xdnn(q_img, q_shape,args):

  # Setup the environment
  img_paths = xdnn_io.getFilePaths(args['images'])
  if(args['golden'] or args['visualize']):
    assert args['labels'], "Provide --labels to compute mAP."
    assert args['results_dir'], "For accuracy measurements, provide --results_dir to save the detections."
    labels = xdnn_io.get_labels(args['labels'])
    colors = generate_colors(len(labels))

  if   args['yolo_version'] == 'v2': yolo_postproc = yolo.yolov2_postproc
  elif args['yolo_version'] == 'v3': yolo_postproc = yolo.yolov3_postproc

  runner = Runner(args['vitis_rundir'])

  # Setup the blobs
  inTensors = runner.get_input_tensors()
  outTensors = runner.get_output_tensors()
  batch_sz = args['batch_sz']
  if batch_sz == -1:
    batch_sz = inTensors[0].dims[0] 

  fpgaBlobs = []
  for io in [inTensors, outTensors]:
    blobs = []
    for t in io:
      shape = (batch_sz,) + tuple([t.dims[i] for i in range(t.ndims)][1:])
      blobs.append(np.empty((shape), dtype=np.float32, order='C'))
    fpgaBlobs.append(blobs)
  fpgaInput = fpgaBlobs[0][0]

  # Setup the YOLO config
  net_h, net_w = fpgaInput.shape[-2:]
  args['net_h'] = net_h
  args['net_w'] = net_w
  biases = bias_selector(args)

  # Setup profiling env
  prep_time = 0
  exec_time = 0
  post_time = 0

  # Start the execution
  for i in range(0, len(img_paths), batch_sz):
    pl = []
    img_shapes = []
    # Prep images
    t1 = timeit.default_timer()
    for j, p in enumerate(img_paths[i:i + batch_sz]):
      fpgaInput[j, ...], img_shape = q_img.get(),q_shape.get()	  
      pl.append(p)
      img_shapes.append(img_shape)
    t2 = timeit.default_timer()

    # Execute
    jid = runner.execute_async(fpgaBlobs[0], fpgaBlobs[1])
    runner.wait(jid)

    # Post Proc
    t3 = timeit.default_timer()
    boxes = yolo_postproc(fpgaBlobs[1], args, img_shapes, biases=biases)
    t4 = timeit.default_timer()

    prep_time += (t2-t1)
    exec_time += (t3-t2)
    post_time += (t4-t3)

    for i in range(min(batch_sz, len(img_shapes))):
      print("Detected {} boxes in {}".format(len(boxes[i]), pl[i]))

    # Save the result
    if(args['results_dir']):
      for i in range(min(batch_sz, len(img_shapes))):
        filename = os.path.splitext(os.path.basename(pl[i]))[0]
        out_file_txt = os.path.join(args['results_dir'], filename + '.txt')
        print("Saving {} boxes to {}".format(len(boxes[i]), out_file_txt)); sys.stdout.flush()
        saveDetectionDarknetStyle(out_file_txt, boxes[i], img_shapes[i])
        if(args['visualize']):
          out_file_png = os.path.join(args['results_dir'], filename + '.png')
          print("Saving result to {}".format(out_file_png)); sys.stdout.flush()
          draw_boxes(pl[i], boxes[i], labels, colors, out_file_png)

  # Profiling results
  if(args['profile']):
    print("\nAverage Latency in ms:")
    print("  Image Prep: {0:3f}".format(prep_time * 1000.0 / len(img_paths)))
    print("  Exec: {0:3f}".format(exec_time * 1000.0 / len(img_paths)))
    print("  Post Proc: {0:3f}".format(post_time * 1000.0 / len(img_paths)))
    sys.stdout.flush()

  # mAP calculation
  if(args['golden']):
    print()
    print("Computing mAP score  : ")
    print("Class names are  : {} ".format(labels))
    mAP = calc_detector_mAP(args['results_dir'], args['golden'], len(labels), labels, args['prob_threshold'], args['iouthresh'])
    sys.stdout.flush()

if __name__ == '__main__':
  #main()
  parser = xdnn_io.default_parser_args()
  
  parser = yolo_parser_args(parser)
  args = parser.parse_args()
  args = xdnn_io.make_dict_args(args)
  q_img = mp.Queue()
  q_shape = mp.Queue()
		#Creating a process to run HW pre-processing kernel
  p_preprocess = mp.Process(target=pre_process,args=(q_img, q_shape, args))
		#Process to run XDNN
  p_xdnn = mp.Process(target=process_xdnn,args=(q_img, q_shape, args))

  p_preprocess.start()
  p_xdnn.start()
  p_preprocess.join()
  p_xdnn.join()

