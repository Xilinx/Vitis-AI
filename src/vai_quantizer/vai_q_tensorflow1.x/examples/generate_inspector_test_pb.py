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
import tensorflow.keras.backend as K
from tensorflow.python.tools import freeze_graph
from tensorflow.keras import layers
from tensorflow.compat.v1.graph_util import convert_variables_to_constants

import os

CKPT_FILE = 'float_model.ckpt'
INFER_FILE = 'inference_graph.pb'
FREEZE_FILE = 'freeze.pb'

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="0", # specify GPU number
        allow_growth=True
    )
)

input_name = 'Input_image_2'
output_name = "final_output"
# inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, 128, 128 , 3], name=input_name)
# net = tf.compat.v1.layers.conv2d(inputs=inputs, filters=16, kernel_size=36, strides=1, use_bias=False, padding='same')
# net = tf.compat.v1.layers.batch_normalization(inputs=net, training=False)
# net = tf.keras.layers.DepthwiseConv2D(kernel_size=3)(net)
# net = tf.compat.v1.layers.conv2d(inputs=net, filters=32, kernel_size=3, strides=1, use_bias=False, padding='same')
# net = tf.compat.v1.layers.batch_normalization(inputs=net, training=False)
# net = tf.nn.relu(net)
# net = tf.compat.v1.layers.batch_normalization(inputs=net, training=False)
# net = tf.nn.relu(net)
# net = tf.nn.selu(net, name="output")
# net = tf.math.multiply(net, 3.)
# net = tf.nn.relu(net, name="relu_2")

K.set_learning_phase(0)

inputs = tf.keras.Input(shape=[128, 128,3], name=input_name)
net = layers.Conv2D(16, 3, strides=1, padding="same", use_bias=False)(inputs)
net = layers.BatchNormalization()(net)
net = tf.nn.relu(net)
net = layers.Conv2D(16, 32, strides=1, padding="same", use_bias=True)(net)
net = tf.nn.relu(net)
net = layers.DepthwiseConv2D(3, strides=1, padding="same", use_bias=False)(net)
net = layers.BatchNormalization()(net)
net = tf.nn.relu(net)
net = layers.DepthwiseConv2D(32, strides=1, padding="same", use_bias=True)(net)
net = tf.nn.relu(net)
net = layers.Conv2D(16, 3, strides=1, padding="same", use_bias=False)(net)
net = tf.nn.selu(net, name="output")
net = net * 2.0
net = tf.nn.relu(net, name=output_name)
output_names = [output_name]


with tf.compat.v1.Session(config=config) as sess:
    # print(sess)
    sess.run(tf.compat.v1.initializers.global_variables())
    for n in sess.graph_def.node:
        print(n.name)
    freeze_graph = convert_variables_to_constants(sess, sess.graph_def, output_names)
    tf.io.write_graph(freeze_graph, './', FREEZE_FILE, as_text=False)
    print(' Freeze graph to %s' % FREEZE_FILE)
