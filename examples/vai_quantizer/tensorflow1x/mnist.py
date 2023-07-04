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
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 28, 28, 1], name="input")
y = tf.placeholder(tf.float32, [None, 10])
conv1 = tf.layers.conv2d(x,
                         filters=32,
                         kernel_size=5,
                         strides=1,
                         padding='same')
conv1_bn = tf.layers.batch_normalization(conv1)
conv1_relu = tf.nn.relu(conv1_bn)
depthwise_filter = tf.Variable(tf.random_normal([5, 5, 32, 1]))
conv2 = tf.nn.depthwise_conv2d(conv1_relu,
                               depthwise_filter,
                               strides=[1, 1, 1, 1],
                               padding='SAME')
conv2_bn = tf.layers.batch_normalization(conv2)
conv2_relu = tf.nn.relu(conv2_bn)
flatten = tf.layers.flatten(conv2_relu)
y_conv = tf.layers.dense(flatten, 10)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        batch_x = batch_x.reshape((-1, 28, 28, 1))
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_x, y: batch_y})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch_x, y: batch_y})

    print('test accuracy %g' % accuracy.eval(
        feed_dict={
            x: np.reshape(mnist.test.images, (-1, 28, 28, 1)),
            y: mnist.test.labels
        }))

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph_def, output_node_names=['dense/BiasAdd'])
    with tf.gfile.FastGFile("float.pb", mode="wb") as f:
        f.write(output_graph_def.SerializeToString())
