# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order
from tensorflow.keras.utils import Sequence
import cv2
import numpy as np

import imagenet_preprocessing

DEFAULT_IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_CLASSES = 1001

NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

NUM_TRAIN_FILES = 1024
SHUFFLE_BUFFER = 10000

DATASET_NAME = 'ImageNet'


def synth_input_fn(is_training, batch_size, height=224, width=224,
        num_channels=3, num_classes=1000, dtype=tf.float32):
  """Returns an input function that returns a dataset with random data.

  This input_fn returns a data set that iterates over a set of random data and
  bypasses all preprocessing, e.g. jpeg decode and copy. The host to device
  copy is still included. This used to find the upper throughput bound when
  tunning the full input pipeline.

  Args:
    height: Integer height that will be used to create a fake image tensor.
    width: Integer width that will be used to create a fake image tensor.
    num_channels: Integer depth that will be used to create a fake image tensor.
    num_classes: Number of classes that should be represented in the fake labels
      tensor
    dtype: Data type for features/images.

  Returns:
    An input_fn that can be used in place of a real one to return a dataset
    that can be used for iteration.
  """
  # pylint: disable=unused-argument
  """Returns dataset filled with random data."""
  # Synthetic input should be within [0, 255].
  inputs = tf.random.truncated_normal(
      [batch_size] + [height, width, num_channels],
      dtype=dtype,
      mean=127,
      stddev=60,
      name='synthetic_inputs')

  labels = tf.random.uniform(
      [batch_size],
      minval=0,
      maxval=num_classes - 1,
      dtype=tf.int32,
      name='synthetic_labels')
  data = tf.data.Dataset.from_tensors((inputs, labels)).repeat()
  data = data.prefetch(buffer_size=1024)
  return data


################################################################################
# Functions for input processing.
################################################################################
def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, num_epochs=1, num_gpus=None,
                           examples_per_epoch=None, dtype=tf.float32):
  """Given a Dataset with raw records, return an iterator over the records.

  Args:
    dataset: A Dataset representing raw records
    is_training: A boolean denoting whether the input is for training.
    batch_size: The number of samples per batch.
    shuffle_buffer: The buffer size to use when shuffling records. A larger
      value results in better randomness, but smaller values reduce startup
      time and use less memory.
    parse_record_fn: A function that takes a raw record and returns the
      corresponding (image, label) pair.
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.
    examples_per_epoch: The number of examples in an epoch.
    dtype: Data type to use for images/features.

  Returns:
    Dataset of (image, label) pairs ready for iteration.
  """

  # We prefetch a batch at a time, This can help smooth out the time taken to
  # load input files as we go through shuffling and processing.
  dataset = dataset.prefetch(buffer_size=batch_size)
  if is_training:
    # Shuffle the records. Note that we shuffle before repeating to ensure
    # that the shuffling respects epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # If we are training over multiple epochs before evaluating, repeat the
  # dataset for the appropriate number of epochs.
  dataset = dataset.repeat(num_epochs)

  if is_training and num_gpus and examples_per_epoch:
    total_examples = num_epochs * examples_per_epoch
    # Force the number of batches to be divisible by the number of devices.
    # This prevents some devices from receiving batches while others do not,
    # which can lead to a lockup. This case will soon be handled directly by
    # distribution strategies, at which point this .take() operation will no
    # longer be needed.
    total_batches = total_examples // batch_size // num_gpus * num_gpus
    dataset.take(total_batches * batch_size)

  # Parse the raw records into images and labels. Testing has shown that setting
  # num_parallel_batches > 1 produces no improvement in throughput, since
  # batch_size is almost always much greater than the number of CPU cores.
  dataset = dataset.apply(
      # tf.contrib.data.map_and_batch(
      tf.data.experimental.map_and_batch(
          lambda value: parse_record_fn(value, is_training, dtype),
          batch_size=batch_size,
          num_parallel_batches=1,
          drop_remainder=False))

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
  # allows DistributionStrategies to adjust how many batches to fetch based
  # on how many devices are present.
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  return dataset


###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if is_training:
    return [
        os.path.join(data_dir, 'train-%05d-of-01024' % i)
        for i in range(NUM_TRAIN_FILES)]
  else:
    return [
        os.path.join(data_dir, 'validation-%05d-of-00128' % i)
        for i in range(128)]


def _parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.

  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields (values are included as examples):

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
  """
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64,
                                              default_value=-1),
      'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
      'image/filename': tf.io.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
  }
  sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.io.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)
  filename = features['image/filename']

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  return features['image/encoded'], label, bbox, filename


def parse_record(raw_record, is_training, dtype, label_offset=1):
  """Parses a record containing a training example of an image.

  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).

  Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.
    dtype: data type to use for images/features.

  Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
  """
  image_buffer, label, bbox, filename = _parse_example_proto(raw_record)
  if label_offset:
    label -= label_offset
  image = imagenet_preprocessing.preprocess_image(
      image_buffer=image_buffer,
      bbox=bbox,
      output_height=DEFAULT_IMAGE_SIZE,
      output_width=DEFAULT_IMAGE_SIZE,
      num_channels=NUM_CHANNELS,
      is_training=is_training,
      convert_bgr=False)
  image = tf.cast(image, dtype)

  # return image, label, filename
  return image, label


def input_fn(is_training, data_dir, batch_size, num_epochs=1, num_gpus=None,
             dtype=tf.float32):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.
    dtype: Data type to use for images/features

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = get_filenames(is_training, data_dir)
  dataset = tf.data.Dataset.from_tensor_slices(filenames)

  if is_training:
    # Shuffle the input files
    dataset = dataset.shuffle(buffer_size=NUM_TRAIN_FILES)

  # Convert to individual records.
  # cycle_length = 10 means 10 files will be read and deserialized in parallel.
  # This number is low enough to not cause too much contention on small systems
  # but high enough to provide the benefits of parallelization. You may want
  # to increase this number if you have a large number of CPU cores.

  dataset = dataset.apply(tf.data.experimental.parallel_interleave(
      tf.data.TFRecordDataset, cycle_length=10))
  # dataset = dataset.apply(tf.contrib.data.parallel_interleave(
  #     tf.data.TFRecordDataset, cycle_length=10))

  return process_record_dataset(
      dataset=dataset,
      is_training=is_training,
      batch_size=batch_size,
      shuffle_buffer=SHUFFLE_BUFFER,
      parse_record_fn=parse_record,
      num_epochs=num_epochs,
      num_gpus=num_gpus,
      examples_per_epoch=NUM_IMAGES['train'] if is_training else None,
      dtype=dtype
  )

def get_images_infor_from_file(image_dir, image_list, label_offset):
  with open(image_list, 'r') as fr:
    lines = fr.readlines()
  imgs = []
  labels = []
  for line in lines:
    img_name, label = line.strip().split(" ")
    img_path = os.path.join(image_dir, img_name)
    label = int(label) + 1 - label_offset
    imgs.append(img_path)
    labels.append(label)
  return imgs, labels

class ImagenetSequence(Sequence):
  def __init__(self, filenames, labels, batch_size, output_height=DEFAULT_IMAGE_SIZE, output_width=DEFAULT_IMAGE_SIZE, central_fraction=0.875):
    self.filenames, self.labels = filenames, labels
    self.batch_size = batch_size
    self.central_fraction = central_fraction
    self.output_height = output_height
    self.output_width = output_width

  def __len__(self):
    return int(np.ceil(len(self.filenames) / float(self.batch_size)))

  def __getitem__(self, idx):
    batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

    processed_imgs = []

    for filename in batch_x:
      # B G R format
      img = cv2.imread(filename)

      # BGR to RGB
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      # Resize
      img = img.astype(float)
      height, width = img.shape[0], img.shape[1]
      new_height = height * 256 // min(img.shape[:2])
      new_width = width * 256 // min(img.shape[:2])
      img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

      # Crop
      height, width = img.shape[0], img.shape[1]
      startx = width//2 - (224//2)
      starty = height//2 - (224//2)
      img = img[starty:starty+224,startx:startx+224]

      # Normalize
      img /= 255.

      # Mean subtraction
      _R_MEAN = 0.485
      _G_MEAN = 0.456
      _B_MEAN = 0.406
      img[..., 0] -= _R_MEAN
      img[..., 1] -= _G_MEAN
      img[..., 2] -= _B_MEAN

      # Std division
      _R_STD = 0.229
      _G_STD = 0.224
      _B_STD = 0.225
      img[..., 0] /= _R_STD
      img[..., 1] /= _G_STD
      img[..., 2] /= _B_STD

      img = cv2.resize(img, (self.output_height, self.output_width), interpolation=cv2.INTER_LINEAR)
      processed_imgs.append(img)
    return np.array(processed_imgs), np.array(batch_y)
