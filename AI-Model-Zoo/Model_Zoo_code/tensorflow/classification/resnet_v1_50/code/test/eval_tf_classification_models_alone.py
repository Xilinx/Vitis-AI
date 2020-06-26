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


# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Provides utilities to preprocess images for the Inception networks."""


import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
import cv2
import numpy as np
import os
import time

tf.app.flags.DEFINE_string(
    'input_graph', 'resnet_v1_18_inference.pb',
    'TensorFlow \'GraphDef\' file to load.')
tf.app.flags.DEFINE_string(
    'eval_image_path', '/home/user/imagenet/val_unresize/',
    'The directory where put the eval images')
tf.app.flags.DEFINE_string(
    'eval_image_list', '/home/user/imagenet/val.txt', 'file has validation images list')
tf.app.flags.DEFINE_string(
    'input_node', 'input', 'input node of pb model')
tf.app.flags.DEFINE_string(
    'output_node', 'resnet_v1_18/predictions/Reshape_1', 'ouput node of pb model')
tf.app.flags.DEFINE_integer(
    'input_height', 224, 'input height of pb model')
tf.app.flags.DEFINE_integer(
    'input_width', 224, 'input width of pb model')
tf.app.flags.DEFINE_integer(
    'label_offset', 1, 'label offset')
tf.app.flags.DEFINE_string(
    'gpus', '0',
    'The gpus used for running evaluation.')

FLAGS = tf.app.flags.FLAGS

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

class Data_loader(object):

  def __init__(self, out_height, out_width, smallest_side=256):
    self._sess = tf.Session()
    self._out_height = out_height
    self._out_width = out_width
    self._smallest_side = smallest_side

    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
     
    self._image_vgg_pl = tf.placeholder(tf.float32, shape=(None, None, 3))
    self._resized_image_vgg = self._aspect_preserving_resize(self._image_vgg_pl, self._smallest_side)

    self._image_inception_pl = tf.placeholder(tf.float32, shape=(None, None, 3))
    self._resized_image_inception = self._inception_central_crop_reisze(self._image_inception_pl) 

  def _center_crop(self, image):
    image_height, image_width = image.shape[:2]
    offset_height = (image_height - self._out_height) // 2
    offset_width = (image_width - self._out_width) // 2
    image = image[offset_height:offset_height+self._out_height, offset_width:offset_width+self._out_width,:]
    return image
 
  def _load_image(self, img_path):
    assert os.path.exists(img_path), img_path + ' doesnot exists!'
    image_data = tf.gfile.GFile(img_path, 'rb').read()
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[-1] == 3
    return image

  def _smallest_size_at_least(self, height, width, smallest_side):
    """Computes new shape with the smallest side equal to `smallest_side`.
    
    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.
    
    Args:
      height: an int32 scalar tensor indicating the current height.
      width: an int32 scalar tensor indicating the current width.
      smallest_side: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.
    
    Returns:
      new_height: an int32 scalar tensor indicating the new height.
      new_width: and int32 scalar tensor indicating the new width.
    """
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)
    
    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)
    
    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(tf.rint(height * scale))
    new_width = tf.to_int32(tf.rint(width * scale))
    return new_height, new_width
    
  def _aspect_preserving_resize(self, image, smallest_side):
    """Resize images preserving the original aspect ratio.
    
    Args:
      image: A 3-D image `Tensor`.
      smallest_side: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.
    
    Returns:
      resized_image: A 3-D tensor containing the resized image.
    """
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = self._smallest_size_at_least(height, width, smallest_side)
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                             align_corners=False)
    resized_image = tf.squeeze(resized_image)
    #resized_image.set_shape([None, None, 3])
    return resized_image
  
  def _inception_central_crop_reisze(self, image, central_fraction=0.875, central_crop=True):
    image = image / 255
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    if central_crop and central_fraction:
      image = tf.image.central_crop(image, central_fraction=central_fraction)
    
    if self._out_height and self._out_width:
      # Resize the image to the specified height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [self._out_height, self._out_width],
                                       align_corners=False)
      image = tf.squeeze(image)

    return image
   
  def _vgg_preprocess(self, image):
    assert image is not None, "image cannot be None"
    resized_image = self._sess.run(self._resized_image_vgg, feed_dict={self._image_vgg_pl: image}) 
    image_crop = self._center_crop(resized_image)  
    image = image_crop - [_R_MEAN, _G_MEAN, _B_MEAN]
    return image
 
  def _inception_preprocess(self, image):
    assert image is not None, "image cannot be None"
    resized_image = self._sess.run(self._resized_image_inception, feed_dict={self._image_inception_pl: image})
    #resized_image = resized_image / 255 
    image = (resized_image - 0.5) * 2
    return image

def _parse_input_graph_proto(input_graph, input_binary):
  """Parser input tensorflow graph into GraphDef proto."""
  if not gfile.Exists(input_graph):
    print("Input graph file '" + input_graph + "' does not exist!")
    return -1
  input_graph_def = graph_pb2.GraphDef()
  mode = "rb" if input_binary else "r"
  with gfile.FastGFile(input_graph, mode) as f:
    if input_binary:
      input_graph_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), input_graph_def)
  return input_graph_def

def main(_):
 
  with tf.Graph().as_default() as graph:
    in_image = tf.placeholder(np.float32, shape=(None, None, 3), name='in_image')
    in_label = tf.placeholder(np.int64, shape=(1), name='in_label') 
    preprocess_img = tf.expand_dims(in_image, axis=0)
    in_label_fa = tf.expand_dims(in_label, axis=0)
 
    input_binary = False if 'txt' in FLAGS.input_graph else True
    input_graph_def = _parse_input_graph_proto(FLAGS.input_graph, input_binary)
    _ = importer.import_graph_def(input_graph_def, name="", input_map={FLAGS.input_node + ':0': preprocess_img})
    logits = graph.get_tensor_by_name(FLAGS.output_node + ':0')
    top1, top1_update = tf.metrics.recall_at_k(in_label_fa, logits, 1, name="precision_top1")
    top5, top5_update = tf.metrics.recall_at_k(in_label_fa, logits, 5, name="precision_top5")
     
    with tf.Session() as sess:
      var_list = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="precision")
      vars_initializer = tf.variables_initializer(var_list=var_list)
      sess.run(vars_initializer)
      height, width = FLAGS.input_height, FLAGS.input_width
      data_loader = Data_loader(height, width)
      with open(FLAGS.eval_image_list, 'r') as fr:
        lines = fr.readlines()
      start_t = time.time()
      val_num = 0
      for line in lines:
        val_num += 1
        if val_num % 1000 == 0:
          print('preprocess %d / %d'%(val_num, len(lines)))
        img_path, label = line.strip().split(" ")
        img_path = os.path.join(FLAGS.eval_image_path, img_path)
        label = int(label) + 1 - FLAGS.label_offset
        label = np.array([label], dtype=np.int64)
        image = data_loader._load_image(img_path)
        image = data_loader._vgg_preprocess(image)
        #image = data_loader._inception_preprocess(image)
        sess.run([top1_update, top5_update], feed_dict={in_image: image, in_label: label})
      end_t = time.time()
      top1, top5 = sess.run([top1, top5])
      print('Recall_1 = [%s]' % str(top1))
      print('Recall_5 = [%s]' % str(top5))
      print('Use_time = [%s]' % str(end_t - start_t))

if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
  tf.app.run()
