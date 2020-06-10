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

# PART OF THIS FILE AT ALL TIMES.

import sys
import os
import numpy as np
import datetime
import scipy.misc
import scipy.io
import cv2
import math
import shutil
sys.path.insert(0,"api/")
from landmark_api import LandmarkPredict
from detect_api import Detect
from align_api import Align


import argparse

parser = argparse.ArgumentParser(description='Get Aligned Faces')
parser.add_argument('--caffe_python_path', type=str, default='../../../../../caffe-xilinx/python/',
                    help='your caffe path')
parser.add_argument('--model_face_detection', type=str, default='../../../densebox_640_360/float/trainval.caffemodel',
                    help='face detection model to get faces')
parser.add_argument('--prototxt_face_detection', type=str, default='../../../densebox_640_360/float/test.prototxt',
                    help='face detection model prototxt to get faces')
parser.add_argument('--model_face_landmark', type=str, default='../../../landmark_sun/float/trainval.caffemodel',
                    help='face landmark model to get face landmarks')
parser.add_argument('--prototxt_face_landmark', type=str, default='../../../landmark_sun/float/test.prototxt',
                    help='face landmark model prototxt to get face landmarks')
parser.add_argument('--input_list', type=str, default='input_list.txt',
                    help='Input image list used to get aligned faces')
parser.add_argument('--output_dir', type=str, default='aligned_faces/',
                    help='path to save aligned faces')

args = parser.parse_args()

def get_aligned_face(img,det,landmark_p,align):
  face_rects=det.detect(img)
  
  #for rect in face_rects:
  #    cv2.rectangle(img,(rect[0],rect[1]),(rect[2],rect[3]),(0,255,0),3)
  if len(face_rects)!=1:
    return 0,0
  face_rect=face_rects[0]
  face_rect=[int(e) for e in face_rect]

  landmark=landmark_p.predict(img,face_rect)
  #for k in xrange(0,10, 2):
  #    cv2.circle(img, (landmark[k], landmark[k+1]), 2, (255, 0, 0), 2)
  expand_scale=0.2
  face_w=face_rect[2]-face_rect[0]
  face_h=face_rect[3]-face_rect[1]
  x_offset=max(int(face_rect[0]-face_w*expand_scale),0) 
  y_offset=max(int(face_rect[1]-face_h*expand_scale),0)
  expand_w=int(face_w*(1+2*expand_scale)) 
  expand_h=int(face_h*(1+2*expand_scale))
  face_rect_expand=[x_offset,y_offset,x_offset+expand_w,y_offset+expand_h]
  face_crop=img[face_rect_expand[1]:face_rect_expand[3],face_rect_expand[0]:face_rect_expand[2],:]
  landmark_src=[
          [landmark[0]-x_offset,landmark[1]-y_offset],
          [landmark[2]-x_offset,landmark[3]-y_offset],
          [landmark[4]-x_offset,landmark[5]-y_offset],
          [landmark[6]-x_offset,landmark[7]-y_offset],
          [landmark[8]-x_offset,landmark[9]-y_offset],
          ]   
  aligned_face_crop = align.warp_and_crop_face(face_crop,landmark_src)
  return 1,aligned_face_crop

if __name__=="__main__":
 
  caffe_python_path=args.caffe_python_path
  sys.path.insert(0,caffe_python_path)
  import caffe
  caffe.set_mode_gpu()
  caffe.set_device(0)
 
  det=Detect()
  det_def_path = args.prototxt_face_detection
  det_model_path = args.model_face_detection 
  det.model_init(caffe_python_path,det_model_path,det_def_path)
 
  landmark_p=LandmarkPredict()
  landmark_def_path = args.prototxt_face_landmark 
  landmark_model_path = args.model_face_landmark  
  landmark_p.model_init(caffe_python_path,landmark_model_path,landmark_def_path,96,72)
  
  align=Align()
  data_base_path=""
  fd=open(args.input_list)
  lines=fd.readlines()
  fd.close()
  if not os.path.exists(args.output_dir):
      os.mkdir(args.output_dir)
  for line in lines:
    line=line.strip()
    img_path=line
    if not os.path.exists(img_path):
      continue
    img=cv2.imread(img_path)
    INPUT_HEIGHT=32*int(img.shape[0]/32)
    INPUT_WIDTH=32*int(img.shape[1]/32)
    img=cv2.resize(img,(INPUT_WIDTH,INPUT_HEIGHT))
  
    check_flag,aligned_face_crop=(get_aligned_face(img,det,landmark_p,align))

    cv2.imwrite(args.output_dir + img_path.split('/')[-1],aligned_face_crop)
