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


import numpy as np
import tensorflow as tf
import os
import json
import cv2

from PIL import Image
from argparse import ArgumentParser
from utils_tf import dboxes_R34_coco, Encoder, cocoval, MyEncoder

def parse_args():
    parser = ArgumentParser(description="SSD Resnet-34 evaluation on COCO")
    parser.add_argument('--model', '-m', type=str, 
            default='../../float/resnet34_tf.22.5.nhwc.pb',
            help='path to frozen graph')
    parser.add_argument('--demo-image', '-oi', type=str,
            default='../../data/1.jpg',
            help='demo image')
    parser.add_argument('--output', '-o', type=str,
            default='../../data/demo.jpg',
            help='demo result')
    parser.add_argument('--labelmap', '-l', type=str,
            default='../../data/coco_labelmap.txt',
            help='path to coco labelmap file')
    parser.add_argument('--score-threshold','-t', type=float,
            default=0.5, help='score threshold')
    return parser.parse_args()

COCO_DICT = [ "None", "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",  
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",  
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",  
              "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",  
              "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
              "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",  
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",  
              "sofa", "pottedplant","bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",  
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",  
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

def get_labelmap(labelmap_path):
    lines = open(labelmap_path).readlines()
    lines = list(map(lambda x:int(x.strip()),lines))
    lines.insert(0,0)
    return lines

def preprocess(image): 
    image = image.astype('float32') / 255.
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    image = (image - mean) / std
    return image #image.transpose([2,0,1])

def run_inference_for_demo(graph, args):
  image_path = args.demo_image
  dboxes = dboxes_R34_coco()
  encoder = Encoder(dboxes)
  with graph.as_default():
    with tf.Session() as sess:
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      output_keys = ['ssd1200/py_location_pred','ssd1200/py_cls_pred']
      for key in output_keys:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image:0')
      image = Image.open(image_path).convert("RGB")
      w_ori, h_ori = image.size
      image=np.array(image.resize((1200,1200), Image.BILINEAR))
      image = preprocess(image)
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})
      ploc = output_dict['ssd1200/py_location_pred']
      plabel = output_dict['ssd1200/py_cls_pred']
      loc, label, prob = encoder.decode_batch(ploc, plabel, 0.50, 200,device=0)[0]
      image_demo = cv2.imread(image_path)
      for i in range(prob.shape[0]-1, -1,-1):
        xmin = int(loc[i][0] * w_ori)
        ymin = int(loc[i][1] * h_ori)
        xmax = int(loc[i][2] * w_ori)
        ymax = int(loc[i][3] * h_ori)
        score = prob[i]
        class_coco = COCO_DICT[label[i]]
        if score < args.score_threshold:
          break 
        cv2.rectangle(image_demo, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
        cv2.putText(image_demo, str(class_coco), (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200), 1)
        cv2.putText(image_demo, str(score), (xmin, ymin + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200), 1)
      cv2.imwrite(args.output, image_demo)

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

    run_inference_for_demo(detection_graph, args)
