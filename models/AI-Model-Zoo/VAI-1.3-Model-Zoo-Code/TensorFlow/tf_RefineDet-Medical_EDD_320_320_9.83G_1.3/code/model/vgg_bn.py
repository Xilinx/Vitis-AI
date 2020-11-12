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

from model.blocks import *

class VGG_BN_Backbone(object):

    def __init__(self, data_format='channels_last'):
        super(VGG_BN_Backbone, self).__init__()
        self._data_format = data_format
 
        self._conv1_bn_block = conv_bn_block(2, 64, 3, (1, 1), data_format=self._data_format, name='conv1')
        self._pool1 = tf.layers.MaxPooling2D(2, 2, padding='same', data_format=self._data_format, name='pool1')
        self._conv2_bn_block = conv_bn_block(2, 128, 3, (1, 1), data_format=self._data_format, name='conv2')
        self._pool2 = tf.layers.MaxPooling2D(2, 2, padding='same', data_format=self._data_format, name='pool2')

        self._conv3_bn_block = conv_bn_block(3, 256, 3, (1, 1), data_format=self._data_format, name='conv3')
        self._pool3 = tf.layers.MaxPooling2D(2, 2, padding='same', data_format=self._data_format, name='pool3')
        self._conv4_bn_block = conv_bn_block(3, 512, 3, (1, 1), data_format=self._data_format, name='conv4')
        self._pool4 = tf.layers.MaxPooling2D(2, 2, padding='same', data_format=self._data_format, name='pool4')

        self._conv5_bn_block = conv_bn_block(3, 512, 3, (1, 1), data_format=self._data_format, name='conv5')
        self._pool5 = tf.layers.MaxPooling2D(3, 1, padding='same', data_format=self._data_format, name='pool5')

        self._conv6 = tf.layers.Conv2D(filters=1024, kernel_size=3, strides=1, padding='same', dilation_rate=6,
                                       data_format=self._data_format, activation=tf.nn.relu, use_bias=True,
                                       kernel_initializer=conv_initializer(),
                                       bias_initializer=tf.zeros_initializer(),
                                       name='fc6_m', _scope='fc6_m', _reuse=None)
        self._conv7 = tf.layers.Conv2D(filters=1024, kernel_size=1, strides=1, padding='same',
                                       data_format=self._data_format, activation=tf.nn.relu, use_bias=True,
                                       kernel_initializer=conv_initializer(),
                                       bias_initializer=tf.zeros_initializer(),
                                       name='fc7_m', _scope='fc7_m', _reuse=None)

    def forward(self, inputs, training=False):
        extract_features = [] 
         
        inputs = forward_block(inputs, self._conv1_bn_block, training=training)
        inputs = forward_block(inputs, self._pool1)
        inputs = forward_block(inputs, self._conv2_bn_block, training=training)
        inputs = forward_block(inputs, self._pool2)

        inputs = forward_block(inputs, self._conv3_bn_block, training=training) 
        inputs = forward_block(inputs, self._pool3)
        inputs = forward_block(inputs, self._conv4_bn_block, training=training)
        extract_features.append(inputs)
        inputs = forward_block(inputs, self._pool4)

        inputs = forward_block(inputs, self._conv5_bn_block, training=training)
        inputs = forward_block(inputs, self._pool5)

        inputs = forward_block(inputs, self._conv6, training=training)
        inputs = forward_block(inputs, self._conv7, training=training)
        extract_features.append(inputs)

        return extract_features 
