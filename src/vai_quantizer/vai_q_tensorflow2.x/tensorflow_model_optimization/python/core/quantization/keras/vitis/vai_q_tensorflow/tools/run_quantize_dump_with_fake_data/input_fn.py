import os
import sys

import numpy as np
import tensorflow as tf

from eval_tf_classification_models_alone import DataLoader



def get_config(key=None, default_value=None):
  if not key:
    raise ValueError("Please assign a key.")
  if not default_value:
    raise ValueEror("Please assign a default_value")

  config = os.environ
  if key in config:
    value = config[key]
    print("Get {} from env: {}".format(key, value))
    return value
  else:
    print("Fail to get {} from env, use default value {}".format(
      key, default_value))
    return default_value



preprocess_type = get_config(key="PREPROCESS_TYPE", default_value="inception")
channel_num = int(get_config(key="CHANNEL_NUM", default_value="3"))
calib_image_dir = get_config(key="CALIB_IMAGE_DIR",
                             default_value="/scratch/workspace/dataset/val_datasets/imagenet/val_dataset")
calib_image_list = get_config(key="CALIB_IMAGE_LIST",
                              default_value="/scratch/workspace/dataset/val_datasets/imagenet/calib_list.txt")
input_node = get_config(key="INPUT_NODES",
                              default_value="image")
calib_batch_size = int(get_config(key="CALIB_BATCH_SIZE", default_value=50))
input_height = int(get_config(key="INPUT_HEIGHT", default_value=224))
input_width = int(get_config(key="INPUT_WIDTH", default_value=224))

class Caliber(object):
  def __init__(self,
               calib_image_dir,
               calib_image_list,
               preprocess_type='inception',
               input_height=224,
               input_width=224,
               calib_batch_size=64):
    self.calib_image_dir = calib_image_dir
    self.calib_image_list = calib_image_list
    self.preprocess_type = preprocess_type
    self.input_height = input_height
    self.input_width = input_width
    self.calib_batch_size = calib_batch_size

  def _calib_input(self, iter):
    with tf.Session() as sess:
      images = []
      data_loader = DataLoader(self.input_height, self.input_width)
      image, input_plhd = data_loader.build_preprocess(
        style=self.preprocess_type)
      lines = open(self.calib_image_list).readlines()
      for index in range(0, self.calib_batch_size):
        curline = lines[iter * self.calib_batch_size + index]
        calib_image_name = curline.strip()
        img_path = os.path.join(self.calib_image_dir, calib_image_name)
        image_calib = sess.run(image, feed_dict={input_plhd: img_path})
        # image_calib = np.squeeze(image_calib)
        image_calib = image_calib[0]
        # generate correct shape with specified channel
        if image_calib.shape[-1] != channel_num:
          tmp_image = np.zeros([input_height, input_width, channel_num])
          for i in range(channel_num):
              tmp_image[:, :, i] = image_calib[:, :, i%3]
          image_calib = tmp_image
        images.append(image_calib)
      return {"{}".format(input_node): images}




caliber = Caliber(calib_image_dir, calib_image_list, preprocess_type,
                  input_height, input_width, calib_batch_size)
calib_input = caliber._calib_input




dump_image_dir = get_config(key="DUMP_IMAGE_DIR",
                             default_value="../../data/Imagenet/val_dataset")
dump_image_list = get_config(key="DUMP_IMAGE_LIST",
                              default_value="../../data/calib_list.txt")
dump_batch_size = int(get_config(key="DUMP_BATCH_SIZE", default_value=50))
dumper = Caliber(dump_image_dir, dump_image_list, preprocess_type,
                  input_height, input_width, dump_batch_size)
dump_input = dumper._calib_input
