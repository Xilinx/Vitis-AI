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

from datetime import datetime
import os
import random
import sys
import threading
import xml.etree.ElementTree as xml_tree

import numpy as np
import six
import tensorflow as tf

import dataset_common

'''How to organize coco dataset folder:
 coco2017/
       |->Annotations/
       |->Images/
       |->train2017.txt
       |->val2017.txt
'''
tf.app.flags.DEFINE_string('dataset_directory', '../../data/coco2017/',
                           'All datas directory')
tf.app.flags.DEFINE_string('train_splits', 'train2017',
                           'Comma-separated list of the training data sub-directory')
tf.app.flags.DEFINE_string('validation_splits', 'val2017',
                           'Comma-separated list of the validation data sub-directory')
tf.app.flags.DEFINE_string('output_directory', '../../data/tfrecords',
                           'Output data directory')
tf.app.flags.DEFINE_integer('train_shards', 16,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 16,
                            'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 8,
                            'Number of threads to preprocess the images.')
RANDOM_SEED = 180428
FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_list_feature(value):
    """Wrapper for inserting a list of bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if isinstance(value, six.string_types):
        value = six.binary_type(value, encoding='utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(image_name, image_buffer, bboxes, labels, labels_text,
                        difficult, truncated, height, width):
    """Build an Example proto for an example.
    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image_buffer: string, JPEG encoding of RGB image
      bboxes: List of bounding boxes for each image
      labels: List of labels for bounding box
      labels_text: List of labels' name for bounding box
      difficult: List of ints indicate the difficulty of that bounding box
      truncated: List of ints indicate the truncation of that bounding box
      height: integer, image height in pixels
      width: integer, image width in pixels
    Returns:
      Example proto
    """
    ymin = []
    xmin = []
    ymax = []
    xmax = []
    for bb in bboxes:
        assert len(bb) == 4
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], bb)]
    channels = 3
    image_format = 'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/channels': _int64_feature(channels),
        'image/shape': _int64_feature([height, width, channels]),
        'image/object/bbox/xmin': _float_feature(xmin),
        'image/object/bbox/xmax': _float_feature(xmax),
        'image/object/bbox/ymin': _float_feature(ymin),
        'image/object/bbox/ymax': _float_feature(ymax),
        'image/object/bbox/label': _int64_feature(labels),
        'image/object/bbox/label_text': _bytes_list_feature(labels_text),
        'image/object/bbox/difficult': _int64_feature(difficult),
        'image/object/bbox/truncated': _int64_feature(truncated),
        'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(image_name.encode('utf8')),
        'image/encoded': _bytes_feature(image_buffer)}))
    return example

class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""
    def __init__(self):
        self._sess = tf.Session()
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
 
    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})
 
    def cmyk_to_rgb(self, image_data):
        return self._sess.run(self._cmyk_to_rgb,
                              feed_dict={self._cmyk_data: image_data})
 
    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def _process_image(filename, coder):
    """Process a single image file.
    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    with tf.gfile.FastGFile(filename, 'rb') as f_image:
        image_data = f_image.read()
    image = coder.decode_jpeg(image_data)
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    return image_data, height, width

def _find_image_bounding_boxes(directory, cur_record, height, width):
    """Find the bounding boxes for a given image file.
    Args:
      directory: string; the path of all datas.
      cur_record: list of strings; the first of which is the sub-directory of cur_record, the second is the image filename.
    Returns:
      bboxes: List of bounding boxes for each image.
      labels: List of labels for bounding box.
      labels_text: List of labels' name for bounding box.
      difficult: List of ints indicate the difficulty of that bounding box.
      truncated: List of ints indicate the truncation of that bounding box.
    """
    anna_file = os.path.join(directory, 'Annotations', cur_record[1].replace('jpg', 'txt'))
    assert os.path.exists(anna_file)
    with open(anna_file, 'r') as f_anno:
        anna_lines = f_anno.readlines()
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for anna_line in anna_lines:
        items = anna_line.strip().split(" ")
        if len(items) > 6:
            assert len(items) == 7
            #print(items)
            items[1] = items[1] + ' ' + items[2]
            for ind in range(2, 6):
                items[ind]  = items[ind + 1]
            items = items[0:6]
            #print(items)
        label = items[1]
        labels.append(int(dataset_common.COCO_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))
        difficult.append(0)
        truncated.append(0)
        '''
        bboxes.append((float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]))
        '''
        bboxes.append((float(items[3]) / height,
                       float(items[2]) / width,
                       float(items[5]) / height,
                       float(items[4]) / width))
    return bboxes, labels, labels_text, difficult, truncated

def _process_image_files_batch(coder, thread_index, ranges, name, directory, all_records, num_shards):
    """Processes and saves list of images as TFRecord in 1 thread.
    Args:
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
      thread_index: integer, unique batch to run index is within [0, len(ranges)).
      ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.
      name: string, unique identifier specifying the data set
      directory: string; the path of all datas
      all_records: list of string tuples; the first of each tuple is the sub-directory of the record, the second is the image filename.
      num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)
    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]
    counter = 0
    for s in range(num_shards_per_batch):
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)
        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            cur_record = all_records[i]
            filename = os.path.join(directory, 'Images', cur_record[1])
            #print(filename)
            image_buffer, height, width = _process_image(filename, coder)
            bboxes, labels, labels_text, difficult, truncated = _find_image_bounding_boxes(directory, cur_record, height, width)
            #print(xxx)
            #image_buffer, height, width = _process_image(filename, coder)
            example = _convert_to_example(cur_record[1], image_buffer, bboxes, labels, labels_text,
                                          difficult, truncated, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1
            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()
        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()

def _process_image_files(name, directory, all_records, num_shards):
    """Process and save list of images as TFRecord of Example protos.
    Args:
      name: string, unique identifier specifying the data set
      directory: string; the path of all datas
      all_records: list of string tuples; the first of each tuple is the sub-directory of the record, the second is the image filename.
      num_shards: integer number of shards for this data set.
    """
    spacing = np.linspace(0, len(all_records), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()
    coord = tf.train.Coordinator()
    coder = ImageCoder()
    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, directory, all_records, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(all_records)))
    sys.stdout.flush()

def _process_dataset(name, directory, all_splits, num_shards):
    """Process a complete data set and save it as a TFRecord.
    Args:
      name: string, unique identifier specifying the data set.
      directory: string, root path to the data set.
      all_splits: list of strings, sub-path to the data set.
      num_shards: integer number of shards for this data set.
    """
    all_records = []
    for split in all_splits:
        jpeg_file_path = os.path.join(directory, 'Images')
        file_name = split + '.txt'
        assert os.path.exists(os.path.join(directory, file_name))
        with open(os.path.join(directory, file_name)) as f_in:
            image_name_list = f_in.readlines()
        jpegs = [im_name.strip() + '.jpg' for im_name in image_name_list]
        all_records.extend(list(zip([split] * len(jpegs), jpegs)))
    shuffled_index = list(range(len(all_records)))
    random.seed(RANDOM_SEED)
    random.shuffle(shuffled_index)
    all_records = [all_records[i] for i in shuffled_index]
    #print(all_records)
    _process_image_files(name, directory, all_records, num_shards)
    
def parse_comma_list(args):
    return [s.strip() for s in args.split(',')]

def main(unused_argv):
    assert not FLAGS.train_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
    assert not FLAGS.validation_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with '
        'FLAGS.validation_shards')

    if not os.path.exists(FLAGS.output_directory):
        os.mkdir(FLAGS.output_directory) 
 
    print('Saving results to %s' % FLAGS.output_directory)
    _process_dataset('coco_2017_val', FLAGS.dataset_directory, parse_comma_list(FLAGS.validation_splits), FLAGS.validation_shards)
    _process_dataset('coco_2017_train', FLAGS.dataset_directory, parse_comma_list(FLAGS.train_splits), FLAGS.train_shards)

if __name__ == '__main__':
    tf.app.run()
