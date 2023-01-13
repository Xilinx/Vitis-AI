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
# ==============================================================================
import tempfile
import os

import tensorflow as tf
import numpy as np
import datetime

from tensorflow import keras

# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

from tensorflow_model_optimization.quantization.keras.vitis.layers import vitis_activation

# Define the model architecture.
inputs = keras.layers.Input(shape=(28, 28))
x = keras.layers.Reshape(target_shape=(28, 28, 1))(inputs)
x = keras.layers.Conv2D(
    filters=32, kernel_size=(3, 3), use_bias=True, activation='linear')(
        x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.DepthwiseConv2D(
    kernel_size=(3, 3), use_bias=True, activation='linear')(
        x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dropout(rate=0.1)(x)
x = keras.layers.Dense(10)(x)
predictions = x

model = keras.Model(inputs=inputs, outputs=predictions, name="mnist_model")

#  Train the float model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['sparse_categorical_accuracy'])

log_dir = "logs/float_fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)
model.fit(
    train_images,
    train_labels,
    epochs=5,
    validation_data=(test_images, test_labels))

model.save('float.h5')

# Quantize Finetune
from tensorflow_model_optimization.quantization.keras import vitis_quantize

# q_aware stands for for quantization aware.
quantizer = vitis_quantize.VitisQuantizer(model, '8bit_tqt')
qat_model = quantizer.get_qat_model(
    init_quant=True,
    calib_dataset=train_images[0:10],
    include_cle=True,
    freeze_bn_delay=1000)

qat_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

qat_model.fit(
    train_images,
    train_labels,
    epochs=3,
    validation_data=(test_images, test_labels))

quantized_model = vitis_quantize.VitisQuantizer.get_deploy_model(qat_model)
quantized_model.save('quantized.h5')

# Load Quantized Model
quantized_model = keras.models.load_model('quantized.h5')

# Evaluate Quantized Model
quantized_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['sparse_categorical_accuracy'])
quantized_model.evaluate(test_images, test_labels, batch_size=500)

# Dump Quantized Model
quantizer.dump_model(
    quantized_model, dataset=train_images[0:1], dump_float=True)
