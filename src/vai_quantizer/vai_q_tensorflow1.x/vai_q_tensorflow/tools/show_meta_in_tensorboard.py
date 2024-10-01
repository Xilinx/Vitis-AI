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


import tensorflow as tf
import os

import numpy as np
import tensorflow.contrib.decent_q

# from tensorflow.contrib.nccl.python.ops import nccl_ops
# nccl_ops._maybe_load_nccl_ops_so()

tf.app.flags.DEFINE_string('gpu', '0', 'model folder')
tf.app.flags.DEFINE_integer('port', '6006', 'port number')
tf.app.flags.DEFINE_string('input_meta', '', 'meta_graph file')
FLAGS = tf.app.flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

# We clear devices to allow TensorFlow to control on which device it will load operations
clear_devices = True

# We start a session using a temporary fresh Graph
with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(FLAGS.input_meta, clear_devices=clear_devices)
    for node in sess.graph_def.node:
      print(node.name, node.op)

    writer = tf.summary.FileWriter('./logdir/', sess.graph)

writer.flush()
os.system('tensorboard --logdir ./logdir/ --port {}'.format(FLAGS.port))
