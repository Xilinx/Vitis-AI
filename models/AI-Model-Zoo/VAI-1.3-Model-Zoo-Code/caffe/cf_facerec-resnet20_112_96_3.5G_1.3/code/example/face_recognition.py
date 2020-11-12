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
from numpy import *
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
from recog_api import Recog 


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
parser.add_argument('--model_face_recognition', type=str, default='../../../facerec_resnet20/float/trainval.caffemodel',
                    help='face recognition model')
parser.add_argument('--prototxt_face_recognition', type=str, default='../../../facerec_resnet20/float/test.prototxt',
                    help='face recognition model prototxt')
parser.add_argument('--input_list', type=str, default='input_list.txt',
                    help='Input image list used to get face feature.')
parser.add_argument('--face_ID_database', type=str, default='face_ID_list.txt',
                    help='Face ID images list used to construct face features database')
parser.add_argument('--threshold', type=float, default=0.83, 
                    help="Similarity score threshold. If the current score is greater than the threshold, it is judged as the same                    ID, otherwise it is different ID")
parser.add_argument('--output_results', type=str, default='results.txt',
                    help='path to face recognition results')

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
  
  #Face Detection
  det=Detect()
  det_def_path = args.prototxt_face_detection
  det_model_path = args.model_face_detection 
  det.model_init(caffe_python_path,det_model_path,det_def_path)
  
  #Face landmark(five points)
  landmark_p=LandmarkPredict()
  landmark_def_path = args.prototxt_face_landmark 
  landmark_model_path = args.model_face_landmark  
  landmark_p.model_init(caffe_python_path,landmark_model_path,landmark_def_path,96,72)
  
  #Face alignment
  align=Align()
  
  #Face feature extraction
  recog=Recog()
  recog_def_path = args.prototxt_face_recognition
  recog_model_path =args.model_face_recognition
  recog.model_init(caffe_python_path,recog_model_path,recog_def_path,112,96,False)
 
  #start to create face feature database
  print(">>>>>>start to create face feature database")
  face_IDs = []
  face_feature_database = np.zeros((0,512))
  fd=open(args.face_ID_database)
  lines=fd.readlines()
  fd.close()
  #if not os.path.exists(args.output_dir):
  #    os.mkdir(args.output_dir)
  
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
    if check_flag == 0:
        continue
    face_IDs.append(line.split("/")[-1].split('_0')[0])
    #cv2.imwrite(args.output_dir + img_path.split('/')[-1],aligned_face_crop)
    feature=recog.get_feature(aligned_face_crop)
    feature_norm = feature/np.sqrt(np.dot(feature, feature.T))

    face_feature_database = concatenate((face_feature_database,feature_norm),axis=0)


  #start to extract face feature for input images
  print(">>>>>>start to extract current face feature")
  fd=open(args.input_list)
  lines=fd.readlines()
  fd.close()
  
  face_recognition_results = open(args.output_results, 'w')
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
    if check_flag == 0:
        continue
    #cv2.imwrite(args.output_dir + img_path.split('/')[-1],aligned_face_crop)
    current_face_feature=recog.get_feature(aligned_face_crop)

    #cacluate the similarity score between the current face feature and all feature in the face feature database.
    print(">>>>>>cacluate the similarity score between the current face feature and all feature in the face feature database.")
    current_face_feature_norm = current_face_feature/np.sqrt(np.dot(current_face_feature, current_face_feature.T))
    scores = np.dot(current_face_feature, face_feature_database.T)
    socres = scores.tolist()[0]
    scores_remap = []
    for score in socres:
        scores_remap.append(1.0 / (1 + math.exp(-12.4 * score + 3.763)))
    max_index = scores_remap.index(max(scores_remap))
    max_score = scores_remap[max_index]
    if max_score > args.threshold:
        print("******current face ID is %s******"%(face_IDs[max_index]))
        face_recognition_results.write("the face ID with the name %s is %s\n"%(line.split('/')[-1], face_IDs[max_index])) 
    else:
        print("******current face is not face database, please register in advance******")
        face_recognition_results.write("current face is not face database, please register in advance.\n")





