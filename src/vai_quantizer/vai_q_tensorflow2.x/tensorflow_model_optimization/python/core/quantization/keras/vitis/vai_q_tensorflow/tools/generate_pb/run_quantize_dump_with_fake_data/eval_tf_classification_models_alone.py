import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
#from tensorflow.contrib import decent_q
import cv2
import numpy as np
import os
import time
import progressbar

tf.app.flags.DEFINE_string('input_graph',
        '', 'TensorFlow \'GraphDef\' file to load.')
tf.app.flags.DEFINE_string('eval_image_path',
                           '', 'The directory where put the eval images')
tf.app.flags.DEFINE_string('eval_image_list',
        '/group/modelzoo/val_datasets/imagenet/val_list.txt',
                           'file has validation images list')
tf.app.flags.DEFINE_string(
    'preprocess_type', 'inception',
    'image preprocess type, choices are inception and vgg')
tf.app.flags.DEFINE_string('input_node', '', 'input node of pb model')
tf.app.flags.DEFINE_string('output_node', '', 'ouput node of pb model')
tf.app.flags.DEFINE_integer('input_height', 224, 'input height of pb model')
tf.app.flags.DEFINE_integer('input_width', 224, 'input width of pb model')
tf.app.flags.DEFINE_integer('label_offset', 1, 'label offset')
tf.app.flags.DEFINE_integer('eval_iter', 50000, 'eval iterations')
tf.app.flags.DEFINE_integer('eval_batch', 100, 'eval batch size')
tf.app.flags.DEFINE_string('gpus', '0', 'The gpus used for running evaluation.')
tf.app.flags.DEFINE_boolean('use_quantize', False, 'Flag for wheather using quantize or not.')

FLAGS = tf.app.flags.FLAGS

if FLAGS.use_quantize:
  from tensorflow.contrib import decent_q

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

EFFICIENTNET_MEAN_RGB = [127.0, 127.0, 127.0]
EFFICIENTNET_STDDEV_RGB = [128.0, 128.0, 128.0]
CROP_PADDING = 32

class DataLoader(object):
  def __init__(self, height=224, width=224):
    self.output_height = height
    self.output_width = width
    self.resize_side=256

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

    scale = tf.cond(tf.greater(height, width), lambda: smallest_side / width,
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
    new_height, new_width = self._smallest_size_at_least(
        height, width, smallest_side)
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                             align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image

  def _mean_image_subtraction(self, image, means):
    """Subtracts the given means from each image channel.

    For example:
      means = [123.68, 116.779, 103.939]
      image = _mean_image_subtraction(image, means)

    Note that the rank of `image` must be known.

    Args:
      image: a tensor of size [height, width, C].
      means: a C-vector of values to subtract from each channel.

    Returns:
      the centered image.

    Raises:
      ValueError: If the rank of `image` is unknown, if `image` has a rank other
        than three or if the number of channels in `image` doesn't match the
        number of values in `means`.
    """
    if image.get_shape().ndims != 3:
      raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
      channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)

  def _stddev_image_division(self, image, stddev):
    """derive the given stddev from each image channel.

    For example:
      stddev = [128.0, 128.0, 128.0]
      image = _stddev_image_division(image, stddev)

    Note that the rank of `image` must be known.

    Args:
      image: a tensor of size [height, width, C].
      stddev: a C-vector of values to subtract from each channel.

    Returns:
      the centered image.

    Raises:
      ValueError: If the rank of `image` is unknown, if `image` has a rank other
        than three or if the number of channels in `image` doesn't match the
        number of values in `stddev`.
    """
    if image.get_shape().ndims != 3:
      raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(stddev) != num_channels:
      raise ValueError('len(stddev) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
      channels[i] /= stddev[i]
    return tf.concat(axis=2, values=channels)


  def _efficientnet_preprocess(self, image):
    assert image is not None, "image cannot be None"
    assert self.output_height == self.output_width, "image output_width must be equal with output_height"
    image_size = self.output_height
    shape = tf.shape(image)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
      ((image_size / (image_size + CROP_PADDING)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)), tf.int32)

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack([
      offset_height, offset_width, padded_center_crop_size,
      padded_center_crop_size
    ])

    image = image[offset_height:offset_height + padded_center_crop_size, \
                 offset_width:offset_width + padded_center_crop_size, :]
    image = tf.image.resize_bicubic([image], [image_size, image_size])[0]

    image = self._mean_image_subtraction(image, EFFICIENTNET_MEAN_RGB)
    image = self._stddev_image_division(image, EFFICIENTNET_STDDEV_RGB)
    image = tf.expand_dims(image, 0)
    return image

  def _vgg_preprocess(self, image):
    image = self._aspect_preserving_resize(image, self.resize_side)

    ## central crop
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    offset_height = (image_height - self.output_height) // 2
    offset_width = (image_width - self.output_height) // 2
    image = image[offset_height:offset_height + self.output_height,
                  offset_width:offset_width + self.output_width, :]
    image.set_shape([self.output_height, self.output_width, 3])

    image = tf.to_float(image)
    image = self._mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
    image = tf.expand_dims(image, 0)
    return image

  def _inception_preprocess(self, image, central_fraction=0.875,
                                     central_crop=True):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    if central_crop and central_fraction:
      image = tf.image.central_crop(image, central_fraction=central_fraction)

    if self.output_height and self.output_width:
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image,
                                       [self.output_height, self.output_width],
                                       align_corners=False)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image

  def _build_placeholder(self):
    input_plhd = tf.placeholder(tf.string, shape=(None), name="input_image_path")
    image = tf.io.read_file(input_plhd)
    image = tf.image.decode_jpeg(image, channels=3)
    return image, input_plhd

  def build_preprocess(self, style="inception"):
    image, input_plhd = self._build_placeholder()
    preprocess_map = {"efficientnet": self._efficientnet_preprocess,
              "vgg": self._vgg_preprocess,
              "inception": self._inception_preprocess}
    if style in preprocess_map:
      image = preprocess_map[style](image)
    return image, input_plhd

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
  with tf.Session() as sess:
    loader = DataLoader(FLAGS.input_height, FLAGS.input_width)
    image, input_plhd = loader.build_preprocess(style=FLAGS.preprocess_type)
    in_image = tf.placeholder(tf.float32,
                              shape=(None, FLAGS.input_height,
                                  FLAGS.input_width, 3),
                              name='in_image')
    in_label = tf.placeholder(tf.int64, shape=(None, 1), name='in_label')
    input_binary = False if 'txt' in FLAGS.input_graph else True
    input_graph_def = _parse_input_graph_proto(FLAGS.input_graph, input_binary)
    _ = importer.import_graph_def(
        input_graph_def,
        name="",
        input_map={FLAGS.input_node + ':0': in_image})
    logits = sess.graph.get_tensor_by_name(FLAGS.output_node + ':0')
    top1, top1_update = tf.metrics.recall_at_k(in_label,
                                               logits,
                                               1,
                                               name="precision_top1")
    top5, top5_update = tf.metrics.recall_at_k(in_label,
                                               logits,
                                               5,
                                               name="precision_top5")
    start_t = time.time()
    var_list = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                 scope="precision")
    vars_initializer = tf.variables_initializer(var_list=var_list)
    sess.run(vars_initializer)
    with open(FLAGS.eval_image_list, 'r') as fr:
      lines = fr.readlines()
    progress = progressbar.ProgressBar()
    if FLAGS.eval_iter > len(lines):
      raise ValueError(
          "eval_iter(%d) should be fewer than total image numbers(%d)." %
          (FLAGS.eval_iter, len(lines)))
    eval_steps = np.int64(np.ceil(FLAGS.eval_iter / FLAGS.eval_batch))
    start_t = time.time()
    for i in progress(range(eval_steps)):
      batch_images = []
      batch_labels = []
      for j in range(FLAGS.eval_batch):
        idx = i * FLAGS.eval_batch + j
        #import pdb; pdb.set_trace()
        idx = np.min([idx, FLAGS.eval_iter])
        line = lines[idx]
        img_path, label = line.strip().split(" ")
        img_path = os.path.join(FLAGS.eval_image_path, img_path)
        image_val = sess.run(image, feed_dict={input_plhd:img_path})
        label = int(label) + 1 - FLAGS.label_offset
        label = np.array([label], dtype=np.int64)
        batch_images.append(image_val)
        # import pdb; pdb.set_trace()
        batch_labels.append(label)
      batch_images = np.squeeze(batch_images)
      sess.run([top1_update, top5_update],
               feed_dict={
                   in_image: batch_images,
                   in_label: batch_labels
               })
    end_t = time.time()
    top1_val, top5_val = sess.run([top1, top5])
    print('Recall_1 = [%s]' % str(top1_val))
    print('Recall_5 = [%s]' % str(top5_val))
    print('Use_time = [%s]' % str(end_t - start_t))

if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
  tf.app.run()
