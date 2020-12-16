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



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

BN_MOMENTUM = 0.99
BN_EPSILON = 1e-5
USE_FUSED_BN = True
conv_initializer = tf.glorot_uniform_initializer

class ReLuLayer(tf.layers.Layer):

    def __init__(self, name, **kwargs):
        super(ReLuLayer, self).__init__(name=name, trainable=False, **kwargs)
        self._name = name

    def build(self, input_shape):
        self._relu = lambda x: tf.nn.relu(x, name=self._name)
        self.built = True

    def call(self, inputs):
        return self._relu(inputs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)


def forward_block(inputs, layers_block, training=False):
    
    def forward_module(layer, inputs, training=False):
        if isinstance(layer, tf.layers.BatchNormalization) \
            or isinstance(layer, tf.layers.Dropout):
            return layer.apply(inputs, training=training)
        return layer.apply(inputs)
    
    if not isinstance(layers_block, list):
        layers_block = [layers_block] 
    for layer in layers_block:
        inputs = forward_module(layer, inputs, training=training)
    return inputs


def L2_norm(inputs, data_format='channels_last', name=None):
    with tf.name_scope(name, "L2_norm", [inputs]) as name:
        axis = -1 if data_format == 'channels_last' else 1
        square_sum = tf.reduce_sum(tf.square(inputs), axis, keep_dims=True)
        inputs_inv_norm = tf.rsqrt(tf.maximum(square_sum, 1e-10))
        return tf.multiply(inputs, inputs_inv_norm, name=name)


def conv_block(num_blocks, filters, kernel_size, strides, padding='same',
               data_format='channels_last', use_bias=True,
               name=None, reuse=None):
    with tf.variable_scope(name):
        conv_blocks = []
        for idx in range(1, num_blocks + 1):
            conv_blocks.append(tf.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                                data_format=data_format, activation=tf.nn.relu, use_bias=use_bias,
                                                kernel_initializer=conv_initializer(),
                                                bias_initializer=tf.zeros_initializer(),
                                                name='{}_{}'.format(name, idx), _scope='{}_{}'.format(name, idx),
                                                _reuse=reuse))
        return conv_blocks


def conv_bn_block(num_blocks, filters, kernel_size, strides, padding='same',
                  data_format='channels_last', use_bias=True,
                  name=None, reuse=None):
    with tf.variable_scope(name):
        bn_axis = -1 if data_format == 'channels_last' else 1
        conv_bn_blocks = []
        for idx in range(1, num_blocks + 1):
            conv_bn_blocks.append(tf.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                                   data_format=data_format, activation=None, use_bias=use_bias,
                                                   kernel_initializer=conv_initializer(),
                                                   bias_initializer=tf.zeros_initializer(),
                                                   name='{}_{}'.format(name, idx), _scope='{}_{}'.format(name, idx),
                                                   _reuse=reuse))
            conv_bn_blocks.append(tf.layers.BatchNormalization(axis=bn_axis, momentum=BN_MOMENTUM, 
                                                               epsilon=BN_EPSILON, fused=USE_FUSED_BN, 
                                                               name='{}_bn{}'.format(name, idx), _scope='{}_bn{}'.format(name, idx),
                                                               _reuse=reuse))
            conv_bn_blocks.append(ReLuLayer('{}_relu{}'.format(name, idx), _scope='{}_relu{}'.format(name, idx), _reuse=reuse))
        return conv_bn_blocks


def ssd_conv_block(filters, strides, padding='same',
                   data_format='channels_last', use_bias=True,
                   name=None, reuse=None):
    with tf.variable_scope(name):
        ssd_conv_blocks = []
        ssd_conv_blocks.append(tf.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding=padding,
                                                data_format=data_format, activation=tf.nn.relu, use_bias=use_bias,
                                                kernel_initializer=conv_initializer(),
                                                bias_initializer=None,
                                                name='{}_1'.format(name), _scope='{}_1'.format(name),
                                                _reuse=reuse))
        ssd_conv_blocks.append(tf.layers.Conv2D(filters=filters * 2, kernel_size=3, strides=strides, padding=padding,
                                                data_format=data_format, activation=tf.nn.relu, use_bias=use_bias,
                                                kernel_initializer=conv_initializer(),
                                                bias_initializer=None,
                                                name='{}_2'.format(name), _scope='{}_2'.format(name),
                                                _reuse=reuse))
        return ssd_conv_blocks


def ssd_conv_bn_block(filters, strides, padding='same', 
                      data_format='channels_last', use_bias=True,
                      name=None, reuse=None):
    with tf.variable_scope(name):
        bn_axis = -1 if data_format == 'channels_last' else 1
        ssd_conv_bn_blocks = []
        ssd_conv_bn_blocks.append(tf.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding=padding,
                                                   data_format=data_format, activation=None, use_bias=use_bias,
                                                   kernel_initializer=conv_initializer(),
                                                   bias_initializer=None,
                                                   name='{}_1'.format(name), _scope='{}_1'.format(name),
                                                   _reuse=reuse))
        ssd_conv_bn_blocks.append(tf.layers.BatchNormalization(axis=bn_axis, momentum=BN_MOMENTUM,
                                                               epsilon=BN_EPSILON, fused=USE_FUSED_BN,
                                                               name='{}_bn1'.format(name), _scope='{}_bn1'.format(name),
                                                               _reuse=reuse))
        ssd_conv_bn_blocks.append(ReLuLayer('{}_relu1'.format(name), _scope='{}_relu1'.format(name), _reuse=reuse))
        ssd_conv_bn_blocks.append(tf.layers.Conv2D(filters=filters * 2, kernel_size=3, strides=strides, padding=padding,
                                                   data_format=data_format, activation=None, use_bias=use_bias,
                                                   kernel_initializer=conv_initializer(),
                                                   bias_initializer=None,
                                                   name='{}_2'.format(name), _scope='{}_2'.format(name),
                                                   _reuse=reuse))
        ssd_conv_bn_blocks.append(tf.layers.BatchNormalization(axis=bn_axis, momentum=BN_MOMENTUM, 
                                                               epsilon=BN_EPSILON, fused=USE_FUSED_BN,
                                                               name='{}_bn2'.format(name), _scope='{}_bn2'.format(name),
                                                               _reuse=reuse))
        ssd_conv_bn_blocks.append(ReLuLayer('{}_relu2'.format(name), _scope='{}_relu2'.format(name), _reuse=reuse))
        return ssd_conv_bn_blocks


def transfer_connection_block(padding='same', data_format='channels_last', 
                              use_bias=False, has_deconv_layer=True, 
                              deconv_stride=2, name=None, reuse=None):
    with tf.variable_scope(name):  
        transfer_connection_blocks = []
        if has_deconv_layer:
            deconv_connection_blocks = []
        for idx in range(3):
            transfer_connection_blocks.append(tf.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding=padding,
                                                               data_format=data_format, activation=None, use_bias=use_bias,
                                                               kernel_initializer=conv_initializer(),
                                                               bias_initializer=None,
                                                               name='{}_{}'.format(name, idx + 1), _scope='{}_{}'.format(name, idx + 1),
                                                               _reuse=reuse))
            transfer_connection_blocks.append(ReLuLayer('{}_relu{}'.format(name, idx + 1), _scope='{}_relu{}'.format(name, idx + 1), _reuse=reuse))
        if has_deconv_layer:
            deconv_connection_blocks.append(tf.layers.Conv2DTranspose(filters=256, kernel_size=4, strides=deconv_stride, padding=padding,
                                                                      data_format=data_format, activation=None, use_bias=use_bias, 
                                                                      kernel_initializer=conv_initializer(),
                                                                      bias_initializer=None, 
                                                                      name='{}_upsample'.format(name), _scope='{}_upsample'.format(name),
                                                                      _reuse=reuse))
        return {'tcb': transfer_connection_blocks, 'upsmaple': deconv_connection_blocks} if has_deconv_layer else {'tcb': transfer_connection_blocks}     
