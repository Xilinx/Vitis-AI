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

from progressbar import ProgressBar
from functools import partial
import numpy as np
import tensorflow as tf

from vai.dpuv1.rt.xdnn_util import make_list
from vai.dpuv1.rt.xdnn_io import loadImageBlobFromFileScriptBase


########################################################################
## USER EDITABLE:
########################################################################
### Minimum required variables to perform preprocessing
INPUT_NODES  = 'data'
LABEL_OFFSET = 0
BATCH_SIZE   = 1    # batch size for quantization.
                    # batch size for inference (CPU or FPGA) is set from run.py

### Preprocessing formulas
### Available transformations: ['resize', 'resize2mindim', 'resize2maxdim', 'crop_letterbox',
###                             'crop_center', 'crop_central', 'plot', 'pxlscale', 'meansub', 'chtranspose', 'chswap']

CMD_SEQ = {
# for resnet50, inception_v1
    'resnet50': [
                  ('meansub', [103.939, 116.779, 123.68]),
                  ('resize2mindim', [224, 224]),
                  ('crop_center', [224, 224]),
                 ],
    'inception_v1': [
                  ('meansub', [103.939, 116.779, 123.68]),
                  ('resize2mindim', [224, 224]),
                  ('crop_center', [224, 224]),
                 ],
# for inception_v3, inception_v4
    'inception_v3': [
                   ('pxlscale', 1/255.),
                   ('crop_central', [0.875, 0.875]),
                   ('resize', [299, 299]),
                   ('meansub', 0.5),
                   ('pxlscale', 2),
                  ],
    'inception_v4': [
                   ('pxlscale', 1/255.),
                   ('crop_central', [0.875, 0.875]),
                   ('resize', [299, 299]),
                   ('meansub', 0.5),
                   ('pxlscale', 2),
                  ],
# for resnet_v1_101/152
     'resnet_v1_101' : [
                   ('resize2mindim', [256, 256]),
                   ('crop_center', [224, 224]),
                   ('meansub', [103.939, 116.779, 123.68]),
                   ('chswap', [2,1,0]),
                  ],
     'resnet_v1_152' : [
                   ('resize2mindim', [256, 256]),
                   ('crop_center', [224, 224]),
                   ('meansub', [103.939, 116.779, 123.68]),
                   ('chswap', [2,1,0]),
                  ],
# for squeezenet
    'squeezenet': [
                   ('resize2mindim', [227, 227]),
                   ('crop_center', [227, 227]),
                   ('meansub', [104.006, 116.669, 122.679]),
                  ],
}
########################################################################





# Environment Variables (obtained by running "source overlaybins/setup.sh")
import os
IMAGEDIR   = os.environ["HOME"] + "/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/"
IMAGELIST  = os.environ["HOME"] + "/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val.txt"
LABELSLIST = os.environ["HOME"] + "/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/synset_words.txt"

INCLUDE_LABELS = False

def load_image(iter, image_list = None, pre_process_function = None, input_nodes = 'data', get_shapes = False, include_labels = False, batch_size = 1):
    images = []
    shapes = []
    labels = []
    for i in range(batch_size):
        image_name, label = image_list[iter * batch_size + i].split(' ')
        image, shape = loadImageBlobFromFileScriptBase(image_name, pre_process_function)
        images.append(image)
        shapes.append(shape)
        labels.append(int(label))
    if include_labels:
        input_dict = {input_nodes: images, 'labels': labels}
    else:
        input_dict = {input_nodes: images}

    if get_shapes:
        return input_dict, shapes
    else:
        return input_dict

def get_input_fn(pre_processing_function_name, input_nodes, include_labels = False, batch_size = 1):
    cmd_seq = CMD_SEQ[pre_processing_function_name]
    with open(IMAGELIST) as fin:
        lines = fin.readlines()
        image_list = list(map(lambda x:os.path.join(IMAGEDIR, x.strip()), lines))
    return partial(load_image, image_list = image_list, pre_process_function = cmd_seq, input_nodes = input_nodes, include_labels = include_labels, batch_size = batch_size)

def top5_accuracy(graph, input_nodes, output_nodes, iter_cnt, batch_size, pre_processing_function_name, label_offset=0):
  global BATCH_SIZE, INPUT_NODES, INCLUDE_LABELS, LABEL_OFFSET

  INPUT_NODES    = input_nodes
  INCLUDE_LABELS = True
  LABEL_OFFSET   = label_offset
  BATCH_SIZE     = batch_size
  input_fn = get_input_fn(pre_processing_function_name, input_nodes, include_labels = True, batch_size = BATCH_SIZE)

  with tf.Session(graph=graph) as sess:
    input_tensors = {node: sess.graph.get_operation_by_name(node).outputs[0] for node in make_list(input_nodes)}
    output_tensor = sess.graph.get_operation_by_name(output_nodes).outputs[0]

    top1_acc = 0
    top5_acc = 0
    progress = ProgressBar()
    line = open(IMAGELIST).readlines()
    for iter in progress(range(iter_cnt)):
      inputs = input_fn(iter)
      correct_labels = inputs['labels']

      predictions = sess.run(output_tensor, feed_dict={tensor: inputs[name] for name, tensor in input_tensors.items()})

      top1_prediction = np.argmax(predictions, axis=1) - LABEL_OFFSET
      top5_prediction = np.argsort(predictions, axis=1)[:,-5:] - LABEL_OFFSET

      top1_accuracy = sum(top1_prediction == correct_labels)
      top5_accuracy = sum([label in top5_prediction for label in correct_labels])

      top1_acc += top1_accuracy
      top5_acc += top5_accuracy

    total_samples = float(iter_cnt*batch_size)
    final_top1_acc = top1_acc/total_samples
    final_top5_acc = top5_acc/total_samples
    print ('top1_acc:{}, top5_acc:{}'.format(final_top1_acc,final_top5_acc))
    return final_top5_acc, final_top1_acc
