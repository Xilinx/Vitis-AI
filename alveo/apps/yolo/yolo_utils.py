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

import colorsys
import os
import random
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

from vai.dpuv1.rt import xdnn, xdnn_io
from vai.dpuv1.utils.postproc import yolo

def bias_selector(args):
  '''
  Utility to choose bias value based on the network.
  args : command line args passed to test_detect_vitis.py/similar
  return: bias array (np.float32 array)
  '''

  if args['yolo_model'] == 'custom':
    assert args['bias_file'], 'Custom network requires --bias_file to be provided'
    biases = np.loadtxt(args['bias_file']).astype(np.float32).flatten()
  else: # Standard networks
    if 'tiny' in args['yolo_model']:
      biases = yolo.tiny_yolov3_bias_coco
    elif 'v3' in args['yolo_model']:
      biases = yolo.yolov3_bias_coco
    elif 'v2' in args['yolo_model']:
      if args['classes'] == 80:
        biases = yolo.yolov2_bias_coco
      elif args['classes'] == 20:
        biases = yolo.yolov2_bias_voc

  return biases


def saveDetectionDarknetStyle(out_file_txt, bboxes, shapeArr):
  '''
  Write detected boxes to text file in Darknet style, useful for mAP calculation
  out_file_txt : text filename
  bboxes : detected boxes from an image (typically output of yolo.postproc)
  shapeArr : list of imgshape, typically in [C, H, W] format
  '''
  out_line_list = []
  for j in range(len(bboxes)):
    x,y,w,h = darknet_style_xywh(shapeArr[1], shapeArr[0], bboxes[j]["ll"]["x"],bboxes[j]["ll"]["y"],bboxes[j]['ur']['x'],bboxes[j]['ur']['y'])
    line_string = '{} {} {} {} {} {}\n'.format(bboxes[j]["classid"], round(bboxes[j]['prob'],3), x, y, w, h)
    out_line_list.append(line_string)

  with open(out_file_txt, "w") as the_file:
    for line in out_line_list:
      the_file.write(line)


def yolo_parser_args(parser):
  ''' Generate parser YOLO specific arguments '''
  # parser = xdnn_io.default_parser_args()
  parser.add_argument('--deviceID', type = int, default = 0,
                      help='FPGA no. -> FPGA ID to run in case multiple FPGAs')
  parser.add_argument('--scorethresh', type=float, default=0.005,
                      help='thresohold on probability threshold')
  parser.add_argument('--iouthresh', type=float, default=0.3,
                      help='thresohold on iouthresh across 2 candidate detections')
  parser.add_argument('--benchmarkmode', action='store_true',
                      help='bypass pre/post processing for benchmarking')
  parser.add_argument('--profile', action='store_true',
                      help='Print average latencies for preproc/exec/postproc')
  parser.add_argument('--visualize', action='store_true',
                      help='Draw output boxes on the input image and save it to --results_dir')
  parser.add_argument("--yolo_model",  type=str, default='xilinx_yolo_v2',
                      help='Model Name [xilinx_yolo_v2 | tinyyolo_v3_426 | custom')
  parser.add_argument('--in_shape', default=[3,224,224], nargs=3, type=int,
                      help='input dimensions')
  parser.add_argument('--anchorCnt', type=int, default=5,
                      help='Number of anchors')
  parser.add_argument('--results_dir', default=None, type=str, metavar="FILE",
                      help="directory path to store results in darknet style")
  parser.add_argument('--prob_threshold', type=float, default=0.1,
                      help='threshold for calculation of f1 score')
  parser.add_argument('--bias_file', type=str, default=None,
                      help='Text file containing bias values for post-processing')
  parser.add_argument('--classes', type=str, default=80,
                      help='Number of classes to be detected')
  parser.add_argument('--yolo_version', type=str, choices={'v2', 'v3'},
                      help='Which version of yolo: [v2 | v3]')
  parser.add_argument('--numprepproc', type=int, default=4,
                      help='number of parallel processes for feeding images')
  parser.add_argument('--numworkers', type=int, default=4,
                      help='number of parallel processes for inference')
  parser.add_argument('--numstream', type=int, default=16,
                      help='number of FPGA streams')
  return parser

def overlap(x1, w1,  x2, w2):
    w1by2 = w1/2
    w2by2 = w2/2
    left  = max(x1 - w1by2, x2 - w2by2)
    right = min(x1 + w1by2, x2 + w2by2)
    return right - left


def cal_iou(box, truth) :
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    if (w<0 or h<0):
        return 0
    inter_area = w * h
    union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area

    return inter_area * 1.0 / union_area


def sortSecond(val):
    return val[1]

def  apply_nms(boxes, classes, scorethresh, overlap_threshold):

    result_boxes=[]
    box_len = (boxes.shape)[0]
    # print(box_len)
    #print "apply_nms :box shape", boxes.shape
    for k in range(classes):

        key_box_class=[]
        for i in range(box_len):
            key_box_class.append((i, boxes[i,5+k]))

        key_box_class.sort(key = sortSecond, reverse = True)

        exist_box=np.ones(box_len)
        for i in range(box_len):

            box_id = key_box_class[i][0]

            if(exist_box[box_id] == 0):
                continue

            if(boxes[box_id,5 + k] < scorethresh):
                exist_box[box_id] = 0;
                continue

            result_boxes.append([boxes[box_id,0], boxes[box_id,1], boxes[box_id,2], boxes[box_id,3], k, boxes[box_id,5+k]])

            for j in range(i+1, box_len):

                box_id_compare = key_box_class[j][0]
                if(exist_box[box_id_compare] == 0):
                    continue

                overlap = cal_iou(boxes[box_id_compare,:], boxes[box_id,:])
                if(overlap >= overlap_threshold):
                    exist_box[box_id_compare] = 0

    return result_boxes

def sigmoid_ndarray(data_in):
    data_in = -1*data_in
    data_in = np.exp(data_in) + 1
    data_in = np.reciprocal(data_in)

    return data_in

def process_all_yolo_layers(yolo_layers, classes, anchorCnt, nw_in_width, nw_in_height):

    biases =[10,13,16,30,33,23, 30,61,62,45,59,119, 116,90,156,198,373,326]

    scale_feature=[]
    out_yolo_layers=[]
    for output_id in range(len(yolo_layers)):
        scale_feature.append([output_id,yolo_layers[output_id].shape[3]])

    scale_feature.sort(key = sortSecond, reverse = True)

    for output_id in range(len(yolo_layers)):

        out_id_process = scale_feature[output_id][0]
        #print "process_all_yolo_layers :layer shape", out_id_process, yolo_layers[out_id_process].shape
        width  = yolo_layers[out_id_process].shape[3]
        height = yolo_layers[out_id_process].shape[2]
        w_range = np.arange(float(width))
        h_range = np.arange(float(height)).reshape(height,1)


        yolo_layers[out_id_process][:,4::(5+classes),:,:] = sigmoid_ndarray(yolo_layers[out_id_process][:,4::(5+classes),:,:])

        yolo_layers[out_id_process][:,0::(5+classes),:,:] = sigmoid_ndarray(yolo_layers[out_id_process][:,0::(5+classes),:,:])
        yolo_layers[out_id_process][:,1::(5+classes),:,:] = sigmoid_ndarray(yolo_layers[out_id_process][:,1::(5+classes),:,:])
        yolo_layers[out_id_process][:,0::(5+classes),:,:] = (yolo_layers[out_id_process][:,0::(5+classes),:,:] + w_range)/float(width)
        yolo_layers[out_id_process][:,1::(5+classes),:,:] = (yolo_layers[out_id_process][:,1::(5+classes),:,:] + h_range)/float(height)

        yolo_layers[out_id_process][:,2::(5+classes),:,:] = np.exp(yolo_layers[out_id_process][:,2::(5+classes),:,:])
        yolo_layers[out_id_process][:,3::(5+classes),:,:] = np.exp(yolo_layers[out_id_process][:,3::(5+classes),:,:])



        for ankr_cnt in range(anchorCnt):
            channel_number_box_width = ankr_cnt * (5+classes) + 2
            scale_value_width = float(biases[2*ankr_cnt + 2 * anchorCnt * output_id]) /float(nw_in_width)
            channel_number_box_height = ankr_cnt * (5+classes) + 3
            scale_value_height = float(biases[2*ankr_cnt + 2 * anchorCnt * output_id + 1]) /float(nw_in_height)

            yolo_layers[out_id_process][:,channel_number_box_width,:,:] = yolo_layers[out_id_process][:,channel_number_box_width,:,:] * scale_value_width
            yolo_layers[out_id_process][:,channel_number_box_height,:,:] = yolo_layers[out_id_process][:,channel_number_box_height,:,:] * scale_value_height

            channel_number_classes = ankr_cnt * (5+classes) + 5
            channel_number_obj_score = ankr_cnt * (5+classes) + 4

            for class_id in range(classes):
                cur_channel = channel_number_classes + class_id
                yolo_layers[out_id_process][:,cur_channel,:,:] = np.multiply(sigmoid_ndarray(yolo_layers[out_id_process][:,cur_channel,:,:]) , yolo_layers[out_id_process][:,channel_number_obj_score,:,:])

        yolo_layer_shape = yolo_layers[out_id_process].shape
        out_yolo_layers.append(yolo_layers[out_id_process])

    return out_yolo_layers

def darknet_style_xywh(image_width, image_height, llx,lly,urx,ury):
    # Assumes (llx,ury) is upper left corner, and (urx,lly always bottom right
    dw = 1./(image_width)
    dh = 1./(image_height)
    x = (llx + urx)/2.0 - 1
    y = (lly + ury)/2.0 - 1
    w = urx - llx
    h = lly - ury
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return x,y,w,h

# To run evaluation on the MS COCO validation set, we must convert to
# MS COCO style boxes, which although not well explained is
# The X,Y Coordinate of the upper left corner of the bounding box
# Followed by the box W
# And the box H
# ll = lower left
# ur = upper right
def cornersToxywh(llx,lly,urx,ury):
    # Assumes (0,0) is upper left corner, and urx always greater than llx
    w = urx - llx
    x = llx
    # Assumes (0,0) is upper left corner, and lly always greater than ury
    h = lly - ury
    y = ury
    #print("llx: %d, lly: %d, urx %d, ury %d"%(llx,lly,urx,ury))
    #print("Becomes:")
    #print("x: %d, y: %d, w %d, h %d"%(x,y,w,h))
    return x,y,w,h

def softmax(startidx, inputarr, outputarr, n, stride):
    import math

    i = 0
    sumexp = 0.0
    largest = -1*float("inf")

    assert len(inputarr) == len(outputarr), "arrays must be equal"
    for i in range(0, n):
        if inputarr[startidx+i*stride] > largest:
            largest = inputarr[startidx+i*stride]
    for i in range(0, n):
        e = math.exp(inputarr[startidx+i*stride] - largest)
        sumexp += e
        outputarr[startidx+i*stride] = e
    for i in range(0, n):
        outputarr[startidx+i*stride] /= sumexp


def sigmoid(x):
    import math
    #print("Sigmoid called on:")
    #print(x)
    #print("")
    if x > 0 :
        return (1 / (1 + math.exp(-x)))
    else:
        return 1 -(1 / (1 + math.exp(x)))

def generate_colors(classes):
    # added color_file.txt so we can customize bad colors
    fname = "color_file.txt"
    colors = []
    if os.path.isfile(fname) == True:
        cf = open(fname, "r")
        for line in cf:
            line = line.rstrip('\n')
            colors.append(eval(line))
        cf.close()
    else:
        hsv_tuples = [(float(x) / classes, 1., 1.) for x in range(classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.
        cf = open("color_file.txt", "w")
        for i in range(classes):
            print("(%d,%d,%d)"%(colors[i][0],colors[i][1],colors[i][2]), file=cf)
        cf.close

    return colors

def draw_boxes(iimage,bboxes,names,colors=[],outpath="out",fontpath="font",display=False):

    if os.path.isdir('./out') is False:
        os.makedirs('./out')

    image = Image.open(iimage)
    draw = ImageDraw.Draw(image)
    thickness = (image.size[0] + image.size[1]) // 300
    font = ImageFont.truetype(fontpath + '/FiraMono-Medium.otf',15)
    #font = ImageFont.truetype('font/FiraMono-Medium.otf',15)


#    classidset = set()
#    for j in range(len(bboxes)):
#        classidset.add(bboxes[j]['classid'])
#    colorsmap = dict(zip(classidset,range(len(classidset))))
#    colors = generate_colors(len(classidset))

    for j in range(0, len(bboxes)):
        classid = bboxes[j]['classid']
        label   = '{} {:.2f}'.format(names[bboxes[j]['classid']],bboxes[j]['prob'])
        labelsize = draw.textsize(label,font=font)
        for k in range(thickness):
            draw.rectangle([bboxes[j]['ll']['x']+k, bboxes[j]['ll']['y']+k, bboxes[j]['ur']['x']+k, bboxes[j]['ur']['y']+k],outline=colors[classid])
        draw.rectangle([bboxes[j]['ll']['x'], bboxes[j]['ur']['y'], bboxes[j]['ll']['x']+2*thickness+labelsize[0], bboxes[j]['ur']['y']+thickness+labelsize[1]],fill=colors[classid],outline=colors[classid])

    for j in range(0, len(bboxes)):
        classid = bboxes[j]['classid']
        label   = '{} {:.2f}'.format(names[bboxes[j]['classid']],bboxes[j]['prob'])
        labelsize = draw.textsize(label)
        draw.text([bboxes[j]['ll']['x']+2*thickness, bboxes[j]['ur']['y']+thickness], label, font=font)

    del draw

    image.save(outpath,quality=90)

    # DISPLAY BOXES
    if os.getenv('DISPLAY') is not None and display:
      img = cv2.imread(outpath)
      cv2.imshow('boxes',img)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
