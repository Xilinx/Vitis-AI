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


# Copyright 2018 Changan Wang
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

BN_MOMENTUM = 0.99
BN_EPSILON = 1e-5
USE_FUSED_BN = True

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

def forward_module(layer, inputs, training=False):
    if isinstance(layer, tf.layers.BatchNormalization) \
        or isinstance(layer, tf.layers.Dropout):
        return layer.apply(inputs, training=training)
    return layer.apply(inputs)

class Resnet34Backbone(object):

    def __init__(self, data_format='channels_first'):
        super(Resnet34Backbone, self).__init__()
        self._data_format = data_format
        self._bn_axis = -1 if data_format == 'channels_last' else 1
        self._conv_initializer = tf.glorot_uniform_initializer
        self._conv_bn_initializer = tf.glorot_uniform_initializer
        self._conv1_bn_block = self.conv_bn_block(1, 64, 7, (2, 2), 'conv1')
        self._pool1 = tf.layers.MaxPooling2D(3, 2, padding='same', data_format=self._data_format, name='pool1')
        ### stage=1
        self._stage1_residul_block_1 = self.residual_block(64, 3, 'stage1_residul_block1')
        self._stage1_residul_block_2 = self.residual_block(64, 3, 'stage1_residul_block2')
        self._stage1_residul_block_3 = self.residual_block(64, 3, 'stage1_residul_block3')
        self._stage1 = [self._stage1_residul_block_1]
        self._stage1.append(self._stage1_residul_block_2)
        self._stage1.append(self._stage1_residul_block_3)
        ### stage=2
        self._stage2_residul_block_1 = self.residual_block(128, 3, 'stage2_residul_block1', has_pre_stride=True)
        self._stage2_residul_block_2 = self.residual_block(128, 3, 'stage2_residul_block2')     
        self._stage2_residul_block_3 = self.residual_block(128, 3, 'stage2_residul_block3')     
        self._stage2_residul_block_4 = self.residual_block(128, 3, 'stage2_residul_block4')
        self._stage2_downsample = self.conv_bn_block(1, 128, 1, (2, 2), 'stage2_downsample', padding='valid', use_relu=False)
        self._stage2 = [self._stage2_residul_block_1]
        self._stage2.append(self._stage2_residul_block_2)
        self._stage2.append(self._stage2_residul_block_3)
        self._stage2.append(self._stage2_residul_block_4)
        ### stage=3
        self._stage3_residul_block_1 = self.residual_block(256, 3, 'stage3_residul_block1')
        self._stage3_residul_block_2 = self.residual_block(256, 3, 'stage3_residul_block2')     
        self._stage3_residul_block_3 = self.residual_block(256, 3, 'stage3_residul_block3')     
        self._stage3_residul_block_4 = self.residual_block(256, 3, 'stage3_residul_block4')
        self._stage3_residul_block_5 = self.residual_block(256, 3, 'stage3_residul_block5')     
        self._stage3_residul_block_6 = self.residual_block(256, 3, 'stage3_residul_block6')
        self._stage3_downsample = self.conv_bn_block(1, 256, 1, (1, 1), 'stage3_downsample', padding='valid', use_relu=False)
        self._stage3 = [self._stage3_residul_block_1] 
        self._stage3.append(self._stage3_residul_block_2)
        self._stage3.append(self._stage3_residul_block_3)
        self._stage3.append(self._stage3_residul_block_4) 
        self._stage3.append(self._stage3_residul_block_5)
        self._stage3.append(self._stage3_residul_block_6)  
        ### stage=4
        '''
        self._stage4_residul_block_1 = self.residual_block(512, 3, 'stage4_residul_block1')     
        self._stage4_residul_block_2 = self.residual_block(512, 3, 'stage4_residul_block2')     
        self._stage4_residul_block_3 = self.residual_block(512, 3, 'stage4_residul_block3')
        self._stage4 = [self._stage4_residul_block_1]
        self._stage4.append(self._stage4_residul_block_2)
        self._stage4.append(self._stage4_residul_block_3)    
        '''
        self._add_chans = [256, 512, 256, 512, 128, 256, 128, 256, 128, 256]
        with tf.variable_scope('additional_layers') as scope:
            self._conv7_block = self.ssd_conv_block(self._add_chans[0:2], 2, 'conv7', use_bias=True)
            self._conv8_block = self.ssd_conv_block(self._add_chans[2:4], 2, 'conv8', use_bias=True)
            self._conv9_block = self.ssd_conv_block(self._add_chans[4:6], 2, 'conv9', use_bias=True)
            self._conv10_block = self.ssd_conv_block(self._add_chans[6:8], 2, 'conv10', use_bias=True, paddings=['valid', 'valid'])
            self._conv11_block = self.ssd_conv_block(self._add_chans[8:10], 1, 'conv11', use_bias=True, paddings=['valid', 'valid'])

    def l2_normalize(self, inputs, name):
        with tf.name_scope(name, "l2_normalize", [inputs]) as name:
            axis = -1 if self._data_format == 'channels_last' else 1
            square_sum = tf.reduce_sum(tf.square(inputs), axis, keep_dims=True)
            inputs_inv_norm = tf.rsqrt(tf.maximum(square_sum, 1e-10))
            return tf.multiply(inputs, inputs_inv_norm, name=name)

    def forward(self, inputs, training=False):
        feature_layers = []
        for layer in self._conv1_bn_block:
            inputs = forward_module(layer, inputs, training=training)
        inputs = self._pool1.apply(inputs)
        inputs = self.forward_residual_stage(inputs, self._stage1, training=training) 
        inputs = self.forward_residual_stage(inputs, self._stage2, downsample=self._stage2_downsample, training=training)
        inputs = self.forward_residual_stage(inputs, self._stage3, downsample=self._stage3_downsample, training=training)  
        #inputs = self.forward_residual_stage(inputs, self._stage4)
        feature_layers.append(inputs)
        for layer in self._conv7_block:
            inputs = forward_module(layer, inputs, training=training)
        feature_layers.append(inputs)
        for layer in self._conv8_block:
            inputs = forward_module(layer, inputs, training=training)
        feature_layers.append(inputs)
        for layer in self._conv9_block:
            inputs = forward_module(layer, inputs, training=training)
        feature_layers.append(inputs)
        for layer in self._conv10_block:
            inputs = forward_module(layer, inputs, training=training)
        feature_layers.append(inputs)
        for layer in self._conv11_block:
            inputs = forward_module(layer, inputs, training=training)
        feature_layers.append(inputs)
        return feature_layers

    def forward_residual_stage(self, inputs, stage_modules, downsample=None, training=False):
        if downsample is not None:
            downsample_inputs = inputs
            for ind, layer in enumerate(downsample):
                downsample_inputs = forward_module(layer, downsample_inputs, training=training)
        for ins, residual_module in enumerate(stage_modules):
            number_layer = len(residual_module)
            residul_inputs = inputs
            for inr, layer in enumerate(residual_module):
                if ins == 0 and downsample is not None:
                    inputs = forward_module(layer, inputs, training=training)
                    if inr == number_layer - 2:
                        inputs = inputs + downsample_inputs
                else:
                    inputs = forward_module(layer, inputs, training=training)
                    if inr == number_layer - 2:
                        inputs = inputs + residul_inputs
        return inputs

    def residual_block(self, filters, kernel_size, name, padding='same', has_pre_stride=False, reuse=None):    
        if not has_pre_stride:
            residual_blocks = self.conv_bn_block(2, filters, kernel_size, strides=(1, 1), name=name, padding=padding)
        else:
            residual_blocks = self.conv_bn_block(1, filters, kernel_size, strides=(2, 2), name=name+'_1', padding=padding)
            residual_blocks.extend(self.conv_bn_block(1, filters, kernel_size, strides=(1, 1), name=name+'_2', padding=padding))
        return residual_blocks

    def conv_block(self, num_blocks, filters, kernel_size, strides, name, padding='same', reuse=None, use_relu=True, use_bias=False):
        with tf.variable_scope(name):
            conv_blocks = []
            for ind in range(1, num_blocks + 1):
                conv_blocks.append(tf.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                                    data_format=self._data_format, activation=None, use_bias=use_bias,
                                                    kernel_initializer=self._conv_initializer(),
                                                    name='{}_{}'.format(name, ind), _scope='{}_{}'.format(name, ind), _reuse=None))
                if use_relu:
                    conv_blocks.append(ReLuLayer('{}_relu{}'.format(name, ind), _scope='{}_relu{}'.format(name, ind), _reuse=None))
            return conv_blocks

    def conv_bn_block(self, num_blocks, filters, kernel_size, strides, name, padding='same', reuse=None, use_relu=True, use_bias=False):
        with tf.variable_scope(name):
            conv_bn_blocks = []
            for ind in range(1, num_blocks + 1):
                conv_bn_blocks.append(tf.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                                       data_format=self._data_format, activation=None, use_bias=use_bias,
                                                       kernel_initializer=self._conv_initializer(),
                                                       name='{}_{}'.format(name, ind), _scope='{}_{}'.format(name, ind), _reuse=None))
                conv_bn_blocks.append(tf.layers.BatchNormalization(axis=self._bn_axis, momentum=BN_MOMENTUM, epsilon=BN_EPSILON,             
                                                                   fused=USE_FUSED_BN, name='{}_bn{}'.format(name, ind), _scope='{}_bn{}'.format(name, ind), _reuse=None)) 
                if use_relu:
                    conv_bn_blocks.append(ReLuLayer('{}_relu{}'.format(name, ind), _scope='{}_relu{}'.format(name, ind), _reuse=None))
            return conv_bn_blocks

    def ssd_conv_block(self, filters, strides, name, paddings=['valid', 'same'], reuse=None, use_bias=False):
        with tf.variable_scope(name):
            conv_blocks = []
            conv_blocks.append(tf.layers.Conv2D(filters=filters[0], kernel_size=1, strides=1, padding=paddings[0],
                                                data_format=self._data_format, activation=tf.nn.relu, use_bias=use_bias,
                                                kernel_initializer=self._conv_initializer(),
                                                name='{}_1'.format(name), _scope='{}_1'.format(name), _reuse=None))
            conv_blocks.append(tf.layers.Conv2D(filters=filters[1], kernel_size=3, strides=strides, padding=paddings[1],
                                                data_format=self._data_format, activation=tf.nn.relu, use_bias=use_bias,
                                                kernel_initializer=self._conv_initializer(),
                                                name='{}_2'.format(name), _scope='{}_2'.format(name), _reuse=None))
            return conv_blocks

    def ssd_conv_bn_block(self, filters, strides, name, paddings=['valid', 'same'], reuse=None, use_bias=False):
        with tf.variable_scope(name):
            conv_bn_blocks = []
            conv_bn_blocks.append(tf.layers.Conv2D(filters=filters[0], kernel_size=1, strides=1, padding=paddings[0],
                                                   data_format=self._data_format, activation=None, use_bias=use_bias,
                                                   kernel_initializer=self._conv_bn_initializer(),
                                                   name='{}_1'.format(name), _scope='{}_1'.format(name), _reuse=None))
            conv_bn_blocks.append(tf.layers.BatchNormalization(axis=self._bn_axis, momentum=BN_MOMENTUM, epsilon=BN_EPSILON, 
                                                               fused=USE_FUSED_BN, name='{}_bn1'.format(name), _scope='{}_bn1'.format(name), _reuse=None))
            conv_bn_blocks.append(ReLuLayer('{}_relu1'.format(name), _scope='{}_relu1'.format(name), _reuse=None))
            conv_bn_blocks.append(tf.layers.Conv2D(filters=filters[1], kernel_size=3, strides=strides, padding=paddings[1],
                                                   data_format=self._data_format, activation=None, use_bias=use_bias,
                                                   kernel_initializer=self._conv_bn_initializer(),
                                                   name='{}_2'.format(name), _scope='{}_2'.format(name), _reuse=None))
            conv_bn_blocks.append(tf.layers.BatchNormalization(axis=self._bn_axis, momentum=BN_MOMENTUM, epsilon=BN_EPSILON, 
                                                               fused=USE_FUSED_BN, name='{}_bn2'.format(name), _scope='{}_bn2'.format(name), _reuse=None))
            conv_bn_blocks.append(ReLuLayer('{}_relu2'.format(name), _scope='{}_relu2'.format(name), _reuse=None))
            return conv_bn_blocks

def multibox_head(feature_layers, num_classes, num_anchors_depth_per_layer, data_format='channels_first', strides=(1, 1), use_bias=True):
    with tf.variable_scope('multibox_head'):
        cls_preds = []
        loc_preds = []
        for ind, feat in enumerate(feature_layers):
            loc_preds.append(tf.layers.conv2d(feat, num_anchors_depth_per_layer[ind] * 4, (3, 3), use_bias=use_bias,
                                              name='loc_{}'.format(ind), strides=strides,
                                              padding='same', data_format=data_format, activation=None,
                                              kernel_initializer=tf.glorot_uniform_initializer()))
            cls_preds.append(tf.layers.conv2d(feat, num_anchors_depth_per_layer[ind] * num_classes, (3, 3), use_bias=use_bias,
                                              name='cls_{}'.format(ind), strides=strides,
                                              padding='same', data_format=data_format, activation=None,
                                              kernel_initializer=tf.glorot_uniform_initializer()))
        return loc_preds, cls_preds
