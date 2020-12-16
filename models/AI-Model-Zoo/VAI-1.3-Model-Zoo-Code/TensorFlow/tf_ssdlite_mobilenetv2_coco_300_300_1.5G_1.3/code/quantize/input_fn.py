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

import sys
import os
import cv2
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR + '/test')

# import model_utils.preprocess as preprocess
from model_utils.preprocess import PREPROCESS_FUNC
from model_utils.config import CONFIG_MAP

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


model_type = get_config(key="MODEL_TYPE", default_value="ssdlite_mobilenet_v2")
calib_image_dir = get_config(
    key="CALIB_IMAGE_DIR",
    default_value=
    "data/calibration_data/coco_images/")
calib_image_list = get_config(
    key="CALIB_IMAGE_LIST",
    default_value="data/calib_list.txt")
calib_batch_size = int(get_config(key="CALIB_BATCH_SIZE", default_value=1))
input_height = int(get_config(key="INPUT_HEIGHT", default_value=300))
input_width = int(get_config(key="INPUT_WIDTH", default_value=300))


def calib_input(iter):
  images = []
  line = open(calib_image_list).readlines()
  with tf.Graph().as_default():
    raw_image = tf.placeholder(tf.float32,
                               shape=(None, None, None, 3),
                               name="raw_image")
    # preprocess_func = preprocess.PREPROCESS_FUNC[model_type]
    preprocess_func = PREPROCESS_FUNC[CONFIG_MAP[model_type].feature_extractor_type]
    preprocessed_image = preprocess_func(raw_image, input_height, input_width)
    sess = tf.Session()
    for index in range(0, calib_batch_size):
      curline = line[iter * calib_batch_size + index]
      calib_image_name = curline.strip()
      image_path = os.path.join(calib_image_dir, calib_image_name + ".jpg")
      image = cv2.imread(image_path)
      height, width = image.shape[0:2]
      image = image[:, :, ::-1]  # BGR to RGB
      image = np.array(image, dtype=np.float32)
      image = np.expand_dims(image, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name("raw_image:0")
      pre_image = sess.run(preprocessed_image, feed_dict={image_tensor: image})
      pre_image = np.squeeze(pre_image)
      images.append(pre_image)
    sess.close()
  return {"image_tensor": images}

