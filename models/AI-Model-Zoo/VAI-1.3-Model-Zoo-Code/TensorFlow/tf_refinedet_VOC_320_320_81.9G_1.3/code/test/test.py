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

import numpy as np
import tensorflow as tf
import os
import cv2

from argparse import ArgumentParser
from utils_tf import pboxes_vgg_voc, Encoder

def parse_args():
    parser = ArgumentParser(description="Refinedet vgg evaluation on VOC")
    parser.add_argument('--model', '-m', type=str,
            default='../../float/refinedet_vgg.pb',
            help='path to frozen graph')
    parser.add_argument('--data-root', '-d', type=str,
            default='../../data/VOC/images/',
            help='path to validation images')
    parser.add_argument('--image-list', '-i', type=str,
            default='./dataset_config/image_list.txt',
            help='path to image list file')
    parser.add_argument('--output', '-o', type=str,
            default='../../data/refinedet_vgg.txt',
            help='detection output text file path')
    parser.add_argument('--gt_file','-g', type=str,
            default='dataset_config/gt_detection.txt',
            help='path to ground truth annotations')
    parser.add_argument('--score-threshold','-t', type=float,
            default=0.005, help='score threshold')
    parser.add_argument('--compute_map_script','-c', type=str,
            default='evaluation.py', help='compute map script')
    parser.add_argument('--gpus','-gp', type=str,
            default='0', help='gpu id')
    parser.add_argument('--use_quantize','-q', type=bool,
            default=False, help='if eval quantized model')
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

def get_map_dict():
    labels_to_names = {}
    for name, pair in VOC_LABELS.items():
        labels_to_names[pair[0]] = name
    return labels_to_names

def preprocess(image):
    image = image.astype('float32')
    R_MEAN = 123.68
    G_MEAN = 116.78
    B_MEAN = 103.94
    mean = np.array([B_MEAN, G_MEAN, R_MEAN], dtype=np.float32)
    std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    image = (image - mean) / std
    return image

def run_inference_for_eval(graph, args):
  image_root = args.data_root
  image_list_file = args.image_list

  pboxes = pboxes_vgg_voc()
  encoder = Encoder(pboxes)
  map_dict = get_map_dict()

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

      with open(image_list_file, 'r') as f_image:
        image_lines = f_image.readlines()
      f_out = open(args.output, 'w')
      count = 0
      for image_line in image_lines:
        count += 1
        print("process: %d images"%count)
        image_name = image_line.strip()
        image_path = os.path.join(image_root, image_name + ".jpg")
        '''
        image = Image.open(image_path)#.convert("BGR")
        w_ori, h_ori = image.size
        image=np.array(image.resize((320,320), Image.BILINEAR))
        '''
        image = cv2.imread(image_path)
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
          f_out.writelines(' '.join([image_name, map_dict[class_id], str(score), str(xmin), str(ymin), str(xmax), str(ymax)]) + '\n')
      f_out.close()

      cmd = 'python ' + args.compute_map_script + ' -gt_file ' + args.gt_file + ' -result_file ' + args.output + ' -detection_use_07_metric True'
      print(cmd)
      os.system(cmd)

if __name__=='__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.use_quantize:
      from tensorflow.contrib import decent_q

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(args.model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    run_inference_for_eval(detection_graph, args)
