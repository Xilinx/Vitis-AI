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


#coding=utf-8
import tensorflow as tf
import math
import numpy as np
from itertools import product

def center2point(center_x, center_y, width, height):
    return center_x - width / 2., center_y - height / 2., center_x + width / 2., center_y + height / 2.

def point2center(xmin, ymin, xmax, ymax):
    width, height = (xmax - xmin), (ymax - ymin)
    return xmin + width / 2., ymin + height / 2., width, height

def clip(xmin, ymin, xmax, ymax, min_value=0.0, max_value=1.0):
    xmin, ymin = np.clip(xmin, min_value, max_value), np.clip(ymin, min_value, max_value)
    xmax, ymax = np.clip(xmax, min_value, max_value), np.clip(ymax, min_value, max_value)
    return xmin, ymin, xmax, ymax

def to_ltrb_mi(cx, cy, w, h):
    return tf.concat(center2point(cx, cy, w, h), axis=-1)

def to_ltrb(prior_bboxes):
    cx, cy, pw, ph = tf.split(prior_bboxes, 4, axis=-1)
    return to_ltrb_mi(cx, cy, pw, ph)

def to_center_mi(xmin, ymin, xmax, ymax):
    return tf.concat(point2center(xmin, ymin, xmax, ymax), axis=-1)

def to_center(ltrb_bboxes):
    xmin, ymin, xmax, ymax = tf.split(ltrb_bboxes, 4, axis=-1)
    return to_center_mi(xmin, ymin, xmax, ymax)

class Priorbox(object):
 
    def __init__(self, input_shape, feature_shapes, min_sizes, max_sizes, aspect_ratios, steps, offset=0.5, priorbox_order=False, clip=False):
        ''' Priorbox class
        Args:
          input_shape: shape of network input.
          feature_shapes: shapes(h, w) list of detection heads feature map.
          min_sizes: list of min sizes.
          max_sizes: list of max sizes.
          aspect_ratios: list of prior apsect ratios per piexl.
          steps: list of steps
          prior_order: num_prior * h * w if false else h * w * num_prior  
          clip: whether prior need clip or not. 
        '''
        super(Priorbox, self).__init__()
       
        self.input_shape = input_shape
        self.feature_shapes = feature_shapes
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.aspect_ratios = aspect_ratios
        self.steps = steps
        self.offsets = [offset] * len(steps) 
        self.priorbox_order = priorbox_order
        self.clip = clip
        self.num_priors_depth_per_layer = self._compute_num_priors_depth_per_layer()
        
    def __call__(self):
        prior_boxes = []
           
        for idx, feature_shape in enumerate(self.feature_shapes):
            f_h, f_w = feature_shape
            step = self.steps[idx]
            f_h_s = float(self.input_shape[0]) / step[0]
            f_w_s = float(self.input_shape[1]) / step[1]
            offset = self.offsets[idx]
            min_size = self.min_sizes[idx]
            if self.max_sizes:
                max_size = self.max_sizes[idx]
            
            prior_whs_ratios = []
            for size in min_size:
                p_w = float(size) / self.input_shape[1]
                p_h = float(size) / self.input_shape[0]
                prior_whs_ratios.append((p_w, p_h))
            if self.max_sizes:
                for sizes in zip(min_size, max_size):
                    size = math.sqrt(sizes[0] * sizes[1])   
                    p_w = size / self.input_shape[1]
                    p_h = size / self.input_shape[0]
                    prior_whs_ratios.append((p_w, p_h))

            for size in min_size:
                for alpha in self.aspect_ratios[idx]:
                    s_alpha = math.sqrt(alpha)
                    p_w = float(size) / self.input_shape[1]
                    p_h = float(size) / self.input_shape[0]
                    prior_whs_ratios.append((p_w * s_alpha, p_h / s_alpha))
                    prior_whs_ratios.append((p_w / s_alpha, p_h * s_alpha)) 
        
            if not self.priorbox_order:
                for (h, w) in product(range(f_h), range(f_w)):
                    cx = (w + offset) / f_w_s
                    cy = (h + offset) / f_h_s
                    for p_wh in prior_whs_ratios:
                        prior_boxes.append([cx, cy, p_wh[0], p_wh[1]])  
            else:
                for p_wh in prior_whs_ratios:
                    for (h, w) in product(range(f_h), range(f_w)):
                        cx = (w + offset) / f_w_s
                        cy = (h + offset) / f_h_s
                        prior_boxes.append([cx, cy, p_wh[0], p_wh[1]])

            if self.clip:
                 prior_boxes_ltrb = [list(center2point(*prior_box)) for prior_box in prior_boxes]           
                 prior_boxes_ltrb = [list(clip(*prior_box_ltrb)) for prior_box_ltrb in prior_boxes_ltrb]           
                 prior_boxes = [list(point2center(*prior_box_ltrb)) for prior_box_ltrb in prior_boxes_ltrb]
  
        return tf.convert_to_tensor(prior_boxes)         
         
    def _compute_num_priors_depth_per_layer(self):
        num_priors_depth_per_layer = []
        for idx in range(len(self.min_sizes)):
            num_priors = len(self.min_sizes[idx]) + 2 * len(self.min_sizes[idx]) * len(self.aspect_ratios[idx])
            num_priors += 0 if not self.max_sizes else len(self.max_sizes[idx]) 
            num_priors_depth_per_layer.append(num_priors)
        return num_priors_depth_per_layer
 
    def _compute_num_priors_per_layer(self, feature_shapes, num_priors_depth_per_layer):
        pass 

if __name__ == '__main__':
   
   import os
   os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
 
   from config import  priorbox_config 
   priorboxes_obj = Priorbox(priorbox_config.input_shape, priorbox_config.feature_shapes,
                             priorbox_config.min_sizes, priorbox_config.max_sizes,
                             priorbox_config.aspect_ratios, priorbox_config.steps,
                             priorbox_config.offset)
   priorboxes = priorboxes_obj()
   priorboxes = to_ltrb(priorboxes)

   with tf.Session() as sess:
       priorboxes = sess.run(priorboxes)
       for idx in range(priorboxes.shape[0]):
           print(priorboxes[idx]) 
