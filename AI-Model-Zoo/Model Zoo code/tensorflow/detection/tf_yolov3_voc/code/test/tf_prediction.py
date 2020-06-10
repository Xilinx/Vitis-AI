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


# MIT License
# 
# Copyright (c) 2018 qqwweee
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



import tensorflow as tf
import os
import cv2
from tensorflow.python.platform import gfile
import numpy as np
from yolo3_predictor import yolo_predictor
from tqdm import tqdm
import argparse

from configs.model_config import Yolov3VOCConfig as Config

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    ih, iw, _ = image.shape
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_LINEAR)
    new_image = np.ones((h,w,3), np.uint8) * 128
    h_start = (h-nh)//2
    w_start = (w-nw)//2
    new_image[h_start:h_start+nh, w_start:w_start+nw, :] = image
    return new_image


def write_items_to_file(image_id, items, fw):
    for item in items:
        fw.write(image_id + " " + " ".join([str(comp) for comp in item]) + "\n")


def get_detection(image, model_image_size, class_names):
    image_h, image_w, _ = image.shape

    # image preprocessing
    if model_image_size != (None, None):
        assert model_image_size[0]%32 == 0, 'Multiples of 32 required'
        assert model_image_size[1]%32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
    else:
        new_image_size = (image_w - (image_w % 32), image_h - (image_h % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    out_boxes, out_scores, out_classes, out_y = sess.run(
        [pred_boxes, pred_scores, pred_classes, output_y],
        feed_dict={input_x: image_data, input_image_shape: (image_h, image_w)})

    # convert the result to label format
    items = []
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image_h, np.floor(bottom + 0.5).astype('int32'))
        right = min(image_w, np.floor(right + 0.5).astype('int32'))
        item  = [predicted_class, score, left, top, right, bottom]
        items.append(item)

    return items


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-pb_file', type=str, help='path of frozon pb file')
    parser.add_argument('-img_dir', type=str, help='directory of voc test image')
    parser.add_argument('-test_list', type=str, help='path of voc test list')
    parser.add_argument('-result_file', type=str, help='path of voc prediction result')
    FLAGS = parser.parse_args()

    config = Config()
    score_thresh = config.score_thresh
    nms_thresh = config.nms_thresh
    class_names = config.classes
    predictor = yolo_predictor(config)

    sess = tf.Session()
    with gfile.FastGFile(FLAGS.pb_file, 'rb') as f: # file I/O
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read()) # get graph_def from file
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  # import graph
    sess.run(tf.global_variables_initializer())

    input_x = sess.graph.get_tensor_by_name(config.input_node[0])
    output_y1 = sess.graph.get_tensor_by_name(config.output_node[0])
    output_y2 = sess.graph.get_tensor_by_name(config.output_node[1])
    output_y3 = sess.graph.get_tensor_by_name(config.output_node[2])
    output_y = [output_y1, output_y2, output_y3]

    input_image_shape = tf.placeholder(tf.int32, shape=(2))
    pred_boxes, pred_scores, pred_classes = predictor.predict(output_y, input_image_shape)

    with open(FLAGS.test_list) as fr:
            lines = fr.readlines()
    fw = open(FLAGS.result_file, "w")
    for line in tqdm(lines):
        img_id = line.strip()
        img_path = os.path.join(FLAGS.img_dir, img_id + '.jpg')
        image = cv2.imread(img_path)
        image = image[...,::-1] # BGR -> RGB
        items = get_detection(image, (config.height, config.width), class_names)
        write_items_to_file(img_id, items, fw)
    fw.close()
