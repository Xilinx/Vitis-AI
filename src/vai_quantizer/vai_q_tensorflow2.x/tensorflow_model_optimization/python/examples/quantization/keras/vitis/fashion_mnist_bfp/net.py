#!/usr/bin/env python
import gzip
import numpy as np
import os

import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import tensorflow as tf

data_dir = './dataset'

def load_data():
  files = [
          'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
          't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']
  paths = []
  for fname in files:
    paths.append(os.path.join(data_dir, fname))

  with gzip.open(paths[0], 'rb') as lbpath:
    y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(paths[1], 'rb') as imgpath:
    x_train = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

  with gzip.open(paths[2], 'rb') as lbpath:
    y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(paths[3], 'rb') as imgpath:
    x_test = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

  return (x_train, y_train), (x_test, y_test)


num_classes = 10
# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
if os.path.exists(data_dir):
  print('**********************load_data')
  (x_train, y_train), (x_test, y_test) = load_data()
else:
  print('**********************download_data')
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test

# convert class vectors to binary class matrices

tf.config.run_functions_eagerly(True)
def net_fn():
  tf.config.run_functions_eagerly(True)
  inputs = tf.keras.Input(shape=(img_rows, img_cols, 1))  # Returns a placeholder tensor

  # A layer instance is callable on a tensor, and returns a tensor.
  conv1 = layers.Conv2D(filters=32,
                        kernel_size=(3, 3),
                        activation='relu',
                        input_shape=input_shape)(inputs)
  conv2 = layers.Conv2D(64, (3, 3), activation='relu')(conv1)
  conv3 = layers.Conv2D(64, (3, 3), strides=(2,2), activation='relu')(conv2)
  dropout1 = layers.Dropout(0.25)(conv3)
  flat = layers.Flatten()(dropout1)
  dense = layers.Dense(128, activation='relu')(flat)
  dropout2 = layers.Dropout(0.5)(dense)
  predictions = layers.Dense(num_classes, activation='softmax')(dropout2)

  model = tf.keras.Model(inputs=inputs, outputs=predictions)
  return model

def model_fn():
  model = net_fn()
  return tf.keras.estimator.model_to_estimator(model)

def eval_input_fn():
  return tf.estimator.inputs.numpy_input_fn(
      x={"input_1": x_test},
      y=y_test,
      num_epochs=1,
      shuffle=False)

if __name__ == '__main__':
  model = net_fn()
  model.summary()
  print(model.outputs)
  output_names = [out.op.name for out in model.outputs]
  print(output_names)
