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
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
import logging

def get_norm_by_name(name):
    if name == 'batch':
        return tf.keras.layers.BatchNormalization(axis=-1)
    else:
        raise Exception("unknown norm name %s" % name)

def conv(x, filters, kernel_size, strides=1, norm='batch', activation=None, l2=None, rate=1, deconv=False):

    if deconv:
        output_shape = list(K.int_shape(x)[1:])
        output_shape[0] = int(output_shape[0] * strides)
        output_shape[1] = int(output_shape[1] * strides)
        output_shape[2] = filters
        #logger.debug(str(output_shape))

        y = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='SAME', activation=activation,
                                   dilation_rate=rate, kernel_regularizer=regularizers.l2(l2) if l2 else None)(x)
    else:
        y = layers.Conv2D(filters, kernel_size, strides=strides, padding='SAME', activation=activation,
                          kernel_regularizer=regularizers.l2(l2) if l2 else None,
                          dilation_rate=rate)(x)

     # kernel_regularizer=regularizers.l2(l2) if l2 else None
    if norm:
        y = get_norm_by_name(norm)(y)

    y = layers.ReLU()(y)

    return y


def factorized_module(x, dropout=0.3, dilation=[1, 1], l2=None):
    #logger.debug("factorized: %s" % str(locals()))
    n = K.int_shape(x)[-1]
    y = conv(x, n, [3, 1], rate=dilation[0], norm=None, l2=l2)
    y = conv(y, n, [1, 3], rate=dilation[0], l2=l2)
    y = conv(y, n, [3, 1], rate=dilation[1], norm=None, l2=l2)
    y = conv(y, n, [1, 3], rate=dilation[1], l2=l2)
    y = layers.Dropout(dropout)(y)
    y = layers.Add()([x, y])
    return y


def downsample(x, n, activation='relu', norm=None, l2=None):
    #logger.debug('downsample: %s' % str(locals()))
    f_in = int(K.int_shape(x)[-1])
    f_conv = int(n - f_in)
    branch_1 = conv(x, f_conv, 3, strides=2, l2=l2)
    branch_2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)
    return layers.Concatenate(axis=-1)([branch_1, branch_2])


def upsample(x, n, norm=None, activation=None, l2=None):
    return conv(x, n, 3, strides=2, deconv=True, l2=l2)


def erfnet(input_shape=(256, 256, 1), num_classes=3, l2=None):
    x = layers.Input(shape=input_shape, name='inputs')

    y = downsample(x, 16, l2=l2)
    y = downsample(y, 64, l2=l2)

    for i in range(5):
        y = factorized_module(y, dilation=[1, 1], l2=l2)

    y = downsample(y, 128, l2=l2)
    for k in range(2):
        for i in range(4):
            y = factorized_module(y, dilation=[1, pow(2, i + 1)], l2=l2)

    #logger.debug("upsampling...")
    y = upsample(y, 64)
    for i in range(2):
        y = factorized_module(y, dilation=[1, 1], l2=l2)

    y = upsample(y, 16)
    for i in range(2):
        y = factorized_module(y, dilation=[1, 1], l2=l2)

    y = upsample(y, num_classes, l2=l2)
    return Model(inputs=x, outputs=y)


if __name__ == "__main__":
    erfnet().summary()
