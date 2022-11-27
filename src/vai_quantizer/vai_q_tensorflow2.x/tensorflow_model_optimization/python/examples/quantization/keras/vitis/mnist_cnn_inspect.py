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
x = keras.layers.Activation('gelu')(x)
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
    epochs=1,
    validation_data=(test_images, test_labels))

model.save('float.h5')
model.evaluate(test_images, test_labels, batch_size=500)

# Inspect the float model
from tensorflow_model_optimization.quantization.keras import vitis_inspect

# `target` is the target DPU to deploy this model, it can be a name(e.g. "DPUCAHX8L_ISA0"),
# a json(e.g. "./U50/arch.json") or a fingerprint.
inspector = vitis_inspect.VitisInspector(target='DPUCAHX8L_ISA0')

# In this model only `gelu` layer is not supported by DPU target.
# Inspect results will be shown on screen, and more detailed results will be saved in
# 'inspect_results.txt'. We can also visualize the results in 'model.svg'.
inspector.inspect_model(
    model,
    input_shape=[[1, 28, 28, 1]],
    plot=True,
    plot_file='model.svg',
    dump_results=True,
    dump_results_file='inspect_results.txt',
    verbose=0)
