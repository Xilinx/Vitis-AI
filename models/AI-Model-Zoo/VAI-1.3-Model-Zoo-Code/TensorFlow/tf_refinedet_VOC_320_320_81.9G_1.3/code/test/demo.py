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
# bash

import os
import cv2

import numpy as np
import tensorflow as tf

from argparse import ArgumentParser
from utils_tf import pboxes_vgg_voc, Encoder

def parse_args():
    parser = ArgumentParser(description="Refinedet vgg evaluation on VOC")
    parser.add_argument('--model', '-m', type=str, 
            default='../../float/refinedet_vgg.pb',
            help='path to frozen graph')
    parser.add_argument('--demo-image', '-oi', type=str,
            default='../../data/1.jpg',
            help='demo image')
    parser.add_argument('--output', '-o', type=str,
            default='../../data/demo.jpg',
            help='demo result')
    parser.add_argument('--score-threshold','-t', type=float,
            default=0.5, help='score threshold')
    return parser.parse_args()

VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}

colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]


def preprocess(image): 
    image = image.astype('float32')
    R_MEAN = 123.68
    G_MEAN = 116.78 
    B_MEAN = 103.94
    mean = np.array([B_MEAN, G_MEAN, R_MEAN], dtype=np.float32)
    std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    image = (image - mean) / std
    return image 

def run_inference_for_demo(graph, args, image_path):

  pboxes = pboxes_vgg_voc()
  encoder = Encoder(pboxes)

  with graph.as_default():
    with tf.Session() as sess:
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      output_keys = ['arm_cls', 'arm_loc', 'odm_cls', 'odm_loc']
      for key in output_keys:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image:0')
      
      image = cv2.imread(image_path)
      image_ori = image
      h_ori, w_ori = image.shape[:2] 
      image=np.array(cv2.resize(image, (320,320)))
      
      image = preprocess(image)
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})
      arm_cls = output_dict['arm_cls']
      arm_loc = output_dict['arm_loc']
      odm_cls = output_dict['odm_cls']
      odm_loc = output_dict['odm_loc']
      loc, label, prob = encoder.decode_batch(arm_cls, arm_loc, odm_cls, odm_loc, 0.45, 200, device=0)[0]
      valid_mask = np.logical_and((loc[:, 2] - loc[:, 0] > 0), (loc[:, 3] - loc[:, 1] > 0))
      for i in range(prob.shape[0]-1, -1,-1):
        if not valid_mask[i]:
          continue
        score = prob[i]
        if score < args.score_threshold:
          break 
        xmin = loc[i][0] * w_ori
        ymin = loc[i][1] * h_ori
        xmax = loc[i][2] * w_ori
        ymax = loc[i][3] * h_ori
        class_id = label[i]
        print(xmin, ymin, xmax, ymax, class_id, score)
        cv2.rectangle(image_ori, (int(xmin), int(ymin)), (int(xmax), int(ymax)), colors_tableau[int(class_id)], 1)
      cv2.imwrite(args.output, image_ori)        
 
if __name__=='__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(args.model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    image_path = args.demo_image 
    run_inference_for_demo(detection_graph, args, image_path)
