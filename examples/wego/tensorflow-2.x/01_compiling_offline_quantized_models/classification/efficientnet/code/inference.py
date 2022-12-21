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

keras = tf.keras
# Get frozen ConcreteFunction
flags.DEFINE_string('input_graph',
                           '', 'TensorFlow \'h5\' file to load.')
flags.DEFINE_string(
    'eval_image_path', '/scratch/data/Imagenet/val_dataset',
    'The directory where put the eval images')
flags.DEFINE_integer(
    'nthreads', 8, 'thread number')
flags.DEFINE_integer('batch_iter', 2000, 'eval iterations')
flags.DEFINE_string('mode', 'perf', 'normal or perf mode')
FLAGS = flags.FLAGS
filePath="./words.txt"
def run_func():
    r = model(x[0])[0]
    fp = open(filePath, "r")
    data1 = fp.readlines()
    fp.close()
    result = tf.math.top_k(r,5)
    for k in range(5):
        cnt = 0
        for line in data1:
            if cnt == result[1][0][k]:
                print("Top[%d] %f %s" % (k, result[0][0][k], (line.strip)("\n")))
                break
            cnt = cnt + 1
        
def run_thread(cnt):
    for count in range(cnt, n_of_group, FLAGS.nthreads):

        r = model(x[0])[0]
        result = tf.math.top_k(r,5)

def do_run():
    threads = []
    for i in range(FLAGS.nthreads):
        t1 = threading.Thread(target=run_thread, args=(i,))
        threads.append(t1)

    start_t = time.perf_counter()
    for x in threads:
        x.start()
    for x in threads:
        x.join()
    end_t = time.perf_counter()
    return end_t - start_t
if __name__ == '__main__':
    batch = vitis_vai.get_target_info().batch
    model = vitis_vai.create_wego_model(input_h5=FLAGS.input_graph,
            accuracy_mode=vitis_vai.enums.AccuracyMode.ReserveReduantFixNeurons)
    img_paths = get_images_infor_from_file(FLAGS.eval_image_path, 1)
    imagenet_seq = ImagenetSequence(img_paths,  batch)
    x=[]
    n_of_group = FLAGS.batch_iter
    data  = imagenet_seq[0]
    images_batch=[]
    if FLAGS.mode == "normal":
        run_size = 1
    else:
        run_size = batch
    
    for i in range(run_size):
        images_batch.append(data[0])
    x.append(tf.convert_to_tensor(np.array(images_batch),dtype='float32'))
    r = model(x[0])[0]
    if FLAGS.mode == "normal":
        run_func()
    else:
        r_n = 2
        print("[INFO] start perf test..")
        print("[INFO] repeat running %d times with %d images...." %
              (r_n, FLAGS.batch_iter*batch))
        t = 0.0
        for i in range(r_n):
            t += do_run()
        print("=========== Perf Result ==============")
        print("Total Images: %d" % (FLAGS.batch_iter*batch * r_n))
        print('Use_time = [%0.2fs]' % (t))
        print('qps = [%0.2f]' % (float(FLAGS.batch_iter*batch) / (t / r_n)))
     


