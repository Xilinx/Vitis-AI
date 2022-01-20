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

import cv2
import os
import time
import timeit
import threading
from tqdm import tqdm
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
from tensorflow.contrib import vitis_vai
from tensorflow.python.client import timeline
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import dtypes as dtypes_module

from dataloader import DataLoader
from input_fn import calib_input

tf.app.flags.DEFINE_string('input_graph',
                           '', 'TensorFlow \'GraphDef\' file to load.')
tf.app.flags.DEFINE_string('eval_image_path',
                           '', 'The directory where put the eval images')
tf.app.flags.DEFINE_string('eval_image_list',
                           '/workspace/test_performance_classification.list',
                           'file has validation images list')
tf.app.flags.DEFINE_string(
    'preprocess_type', 'inception',
    'image preprocess type, choices are inception and vgg')
tf.app.flags.DEFINE_string('input_node', '', 'input node of pb model')
tf.app.flags.DEFINE_string('output_node', '', 'ouput node of pb model')
tf.app.flags.DEFINE_integer('input_height', 224, 'input height of pb model')
tf.app.flags.DEFINE_integer('input_width', 224, 'input width of pb model')
tf.app.flags.DEFINE_integer('label_offset', 1, 'label offset')
tf.app.flags.DEFINE_integer('eval_iter', 10000, 'eval iterations')
tf.app.flags.DEFINE_integer('eval_batch', 8, 'eval batch size')
tf.app.flags.DEFINE_integer('nthreads', 4, 'threads number')
tf.app.flags.DEFINE_string('mode', '', 'accuracy or perf mode')

FLAGS = tf.app.flags.FLAGS

def make_callable(sess, feed=[], target=[], fetch=[]):

    def name_list_append(src, dist):
        for element in src:
            if isinstance(element, tf.Tensor):
                dist.append(element.op.name)
            elif isinstance(element, tf.Operation):
                dist.append(element.name)
            else:
                raise ValueError("element must be Tensor or Operation")

    callable_opts = config_pb2.CallableOptions()
    name_list_append(feed, callable_opts.feed)
    name_list_append(target, callable_opts.target)
    name_list_append(fetch, callable_opts.fetch)

    callable_object = sess._make_callable_from_options(callable_opts)

    def run_callable(feed_dict):
        feed_values = []
        for key, value in feed_dict.items():
            if not isinstance(value, tf.Tensor):
                key_type = dtypes_module.as_dtype(key.dtype)
                value = np.asarray(value,
                                   dtype=key_type.as_numpy_dtype)
            feed_values.append(value)
        return callable_object(*feed_values)
    return run_callable

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


def run_thread(cnt):
    for count in range(cnt, n_of_group, FLAGS.nthreads):
        # Using callable object rather than sess.run for better performance;
        '''
        sess.run([top1_update, top5_update],
               feed_dict={
                   in_image: batch_group[count],
                   in_label: batch_group_labels[count]
               })
        '''
        sess_callable(feed_dict={
            in_image: batch_group[count],
            in_label: batch_group_labels[count]
        })

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


if __name__ == "__main__":

    sess = tf.compat.v1.Session()

    in_image = tf.compat.v1.placeholder(tf.float32,
                              shape=(None, FLAGS.input_height,
                                     FLAGS.input_width, 3),
                              name='in_image')
    in_label = tf.compat.v1.placeholder(tf.int64, shape=(None, 1), name='in_label')
    input_binary = False if 'txt' in FLAGS.input_graph else True
    input_graph_def = _parse_input_graph_proto(FLAGS.input_graph, input_binary)
    
    # Create wego graph through wego's API
    vai_wego_graph = vitis_vai.create_wego_graph(
        target="DPUCVDX8H_ISA1_F2W2_8PE",
        input_graph_def=input_graph_def)
    
    sess.graph.as_default()
    _ = importer.import_graph_def(
        vai_wego_graph,
        name="",
        input_map={FLAGS.input_node + ':0': in_image})
    logits = sess.graph.get_tensor_by_name(FLAGS.output_node + ':0')
    top1, top1_update = tf.compat.v1.metrics.recall_at_k(in_label,
                                               logits,
                                               1,
                                               name="precision_top1")
    top5, top5_update = tf.compat.v1.metrics.recall_at_k(in_label,
                                               logits,
                                               5,
                                               name="precision_top5")

    var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.LOCAL_VARIABLES,
                                 scope="precision")
    vars_initializer = tf.compat.v1.variables_initializer(var_list=var_list)
    sess.run(vars_initializer)

    with open(FLAGS.eval_image_list, 'r') as fr:
        lines = fr.readlines()
    if FLAGS.eval_iter > len(lines):
        raise ValueError(
            "eval_iter(%d) should be fewer than total image numbers(%d)." %
            (FLAGS.eval_iter, len(lines)))
    eval_steps = np.int64(np.ceil(FLAGS.eval_iter / FLAGS.eval_batch))

    print("[INFO] loading %d images with batch mode..."%(FLAGS.eval_iter))
    batch_group, batch_group_labels = calib_input(FLAGS.preprocess_type, FLAGS.input_height, FLAGS.input_width,
                                                  FLAGS.eval_image_list, eval_steps, FLAGS.eval_batch, FLAGS.eval_iter, 
                                                  FLAGS.eval_image_path, FLAGS.label_offset)

    # Create callable directly for better performance
    sess_callable = make_callable(sess, feed=[in_image, in_label], target=[
                                  top1_update, top5_update])
    n_of_group = len(batch_group)

    mode = FLAGS.mode
    if mode != "accuracy" and mode != "perf":
        raise ValueError(
            "Unsupported mode, support values: [ %s, %s]." %
            ("accuracy", "perf"))

    if mode == "accuracy":
        r_n = 1
        print("[INFO] start accuracy test...")
        do_run()
        top1_val, top5_val = sess.run([top1, top5])
        print("============ Test Result =============")
        print('Total Images: %d' % (FLAGS.eval_iter))
        print('Recall_1 = [%s]' % str(top1_val))
        print('Recall_5 = [%s]' % str(top5_val))
    else:
        r_n = 20
        print("[INFO] start perf test...")
        print("[INFO] repeat running %d times with %d images...." %
              (r_n, FLAGS.eval_iter))
        t = 0.0
        for i in tqdm(range(r_n)):
            t += do_run()
        print("=========== Perf Result ==============")
        print("Total Images: %d" % (FLAGS.eval_iter * r_n))
        print('Use_time = [%0.2fs]' % (t))
        print('qps = [%0.2f]' % (float(FLAGS.eval_iter) / (t / r_n)))

    sess.close()
