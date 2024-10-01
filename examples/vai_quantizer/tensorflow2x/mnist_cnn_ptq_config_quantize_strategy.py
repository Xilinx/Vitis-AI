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
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images / 255.0
test_images = test_images.reshape((-1, 28, 28, 1))

# Define the model architecture.
inputs = keras.layers.Input(shape=(28, 28, 1))
x = keras.layers.Conv2D(
    filters=32,
    kernel_size=(18, 18),
    strides=(1, 1),
    use_bias=False,
    padding='same',
    activation='linear')(
        inputs)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    strides=(5, 5),
    use_bias=False,
    padding='same',
    activation='linear')(
        x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(10)(x)
predictions = x

model = keras.Model(inputs=inputs, outputs=predictions, name="mnist_cnn")
model.summary()

#  Train the digit classification model
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
    epochs=1,
    validation_data=(test_images, test_labels))
model.save('float.h5')

# Quantize
from tensorflow_model_optimization.quantization.keras import vitis_quantize

# Build Quantizer
quantizer = vitis_quantize.VitisQuantizer(model)

# Dump the quantize strategy configs
quantize_strategy = quantizer.dump_quantize_strategy(
    dump_file='./quantize_strategy_v2.json', verbose=2)

# Modify the configs in 'quantize_strategy_v2.json'

# Set the modified quantize strategy configs
quantizer.set_quantize_strategy('./quantize_strategy_v2.json')

# Quantize
quant_model = quantizer.quantize_model(calib_dataset=train_images[0:10])
quant_model.save('quantized_0.h5')
