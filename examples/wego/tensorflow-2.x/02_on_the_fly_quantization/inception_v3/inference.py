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

from tensorflow.compiler import vitis_vai
from dataset import get_images_infor_from_file, ImagenetSequence
from tensorflow.compat.v1 import flags
import tensorflow as tf
import numpy as np
import threading
import time

flags.DEFINE_string('input_graph', '', 'TensorFlow \'h5\' file to load.')
flags.DEFINE_string('eval_image_path', '/scratch/data/Imagenet/val_dataset', 'The directory where put the eval images')
flags.DEFINE_string('calibration_images_folder','','image path for calibration')
flags.DEFINE_integer('calib_batch_size', 50, 'batch size for calibration')
flags.DEFINE_integer('calib_size', 1000, 'images used for calibration')
flags.DEFINE_boolean('serialize',False,'define whether to do serialize')
flags.DEFINE_string('serialized_model_path', '', 'path to store serialized model')
FLAGS = flags.FLAGS

def run_inference(model, img):
    fp = open("./words.txt", "r")
    data = fp.readlines()
    fp.close()
    r = model(img)[0]
    result = tf.math.top_k(r,5)
    for k in range(5):
        cnt = 0
        for line in data:
            if cnt == result[1][0][k]:
                print("Top[%d] %f %s" % (k, result[0][0][k], (line.strip)("\n")))
                break
            cnt = cnt + 1

if __name__ == '__main__':
    
    if FLAGS.serialized_model_path:
        # deserialize and run the wego mod
        saved_model = tf.saved_model.load(FLAGS.serialized_model_path)
        model = saved_model.cube
        print(f"[INFO] successfully loded model from {FLAGS.serialized_model_path}")

    elif FLAGS.input_graph:
        # quantize the float model and compile it with wego api
        calib_paths = get_images_infor_from_file(FLAGS.calibration_images_folder)
        calib_size = len(calib_paths) if len(calib_paths) < FLAGS.calib_size else FLAGS.calib_size
        print("[INFO] will use %d images for calibration." % (calib_size))
        calib_seq = ImagenetSequence(calib_paths[0:calib_size], FLAGS.calib_batch_size)
        new_model = vitis_vai.quantize(FLAGS.input_graph, calib_dataset = calib_seq)
        # compile the model
        print(f"[INFO] compile the quantized model")
        model = vitis_vai.create_wego_model(new_model)
        
        # you can serialize the compiled module to a file
        if FLAGS.serialize:
            print(f"[INFO] serializing compiled wego module")
            to_export = tf.Module()
            to_export.cube = model
            tf.saved_model.save(to_export,"./serialized_model")
            print(f"[INFO] serialization done")

    # Run test
    print(f"[INFO] run test...")
    test_seq = ImagenetSequence(get_images_infor_from_file(FLAGS.eval_image_path), 1)
    test_img = test_seq[0]
    test_img_tensor = tf.convert_to_tensor(np.array(test_img),dtype='float32')
    run_inference(model, test_img_tensor)


