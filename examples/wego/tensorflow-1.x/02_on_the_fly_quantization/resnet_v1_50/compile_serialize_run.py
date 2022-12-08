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

import os
import time
import threading
import numpy as np
from tqdm import tqdm
from config import load_config
from dataloader import DataLoader

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
from tensorflow.contrib import vitis_vai
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import dtypes as dtypes_module

tf.app.flags.DEFINE_string('img_url','', 'source input image')
tf.app.flags.DEFINE_string('config_file','',
                           'model config file.')
tf.app.flags.DEFINE_string('model_path','',
                           'model path.')
tf.app.flags.DEFINE_string(
    'preprocess_type', 'inception',
    'image preprocess type, choices are inception and vgg')
tf.app.flags.DEFINE_string('mode', '', 'normal or perf mode')

FLAGS = tf.app.flags.FLAGS
model_config = load_config(FLAGS.config_file)

def get_categories():
    # https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories

def read_image(image_path):
    with tf.compat.v1.Session() as sess:
        data_loader = DataLoader(model_config['preprocess']['input_size'][0], model_config['preprocess']['input_size'][1])
        image, input_plhd = data_loader.build_preprocess(
            style=model_config['preprocess']['preprocess_type'])
        image_val = sess.run(image, feed_dict={input_plhd: image_path})
    return image_val 

def make_callable(sess, feed=[], target=[], fetch=[]):

    def name_list_append(src, dist):
        for element in src:
            if isinstance(element, tf.Tensor):
                dist.append(element.name)
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

def _parse_input_graph_proto(input_graph):
    """Parser input tensorflow graph into GraphDef proto."""
    if not gfile.Exists(input_graph):
        print("Input graph file '" + input_graph + "' does not exist!")
        return -1
    input_graph_def = graph_pb2.GraphDef()
    with tf.io.gfile.GFile(input_graph, 'rb') as f:
        input_graph_def.ParseFromString(f.read())
    return input_graph_def

def run_thread(images, t_id, repeat_batch, n_threads):
    for j in range(t_id, repeat_batch, n_threads):
        sess_callable(feed_dict={
            in_image: images[j]
            })
def run(images, n_threads,repeat_batch):
    thread_list = []
    for t_id in range(0, n_threads):
        t = threading.Thread(
            target = run_thread,
            args = (images, t_id, repeat_batch, n_threads)
        )
        thread_list.append(t)
    
    st = time.perf_counter()
    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join()
    et = time.perf_counter()
    
    return (et-st)

def run_perf(img):
    # create batch images based on batch size;
    batch_size = vitis_vai.get_target_info().batch
    batch_images = np.concatenate([img for i in range(batch_size)], 0)
    # run images with repeated batch images;
    repeat_batch = int(model_config["preprocess"]["eval_iter"] / batch_size)
    all_images = [batch_images] * repeat_batch

    n_images = repeat_batch * vitis_vai.get_target_info().batch
    print("[Info] warmup...")
    for i in tqdm(range(5)):
      run(all_images, model_config["preprocess"]["threads"],repeat_batch)    
    r_n = 20
    print("[Info] begin to run inference using %d images with %d times." % (n_images, r_n))
    t = 0.0
    for i in tqdm(range(r_n)):
      t_ = run(all_images, model_config["preprocess"]["threads"],repeat_batch)
      t += t_
    print("===================== Perf Result =====================")
    print("[Total Images] %d" % (r_n * n_images))
    print("[Total Time]   %0.6fs" % float(t))
    print("[FPS]          %0.2f" % (float(n_images) / (t / r_n)))

def run_normal(input_img):
    top5_prob, top5_catid, _ = sess_callable(feed_dict={
                    in_image: input_img
                    })
    categories = get_categories()
    print("=====================TopK Result=====================")
    for i in range(len(top5_prob[0])):
        print(categories[int(top5_catid[0][i])], top5_prob[0][i])


if __name__ == "__main__":
    sess = tf.compat.v1.Session()
    in_image = tf.compat.v1.placeholder(tf.float32,
                              shape=(None,model_config['preprocess']['input_size'][0],model_config['preprocess']['input_size'][1],3),
                              name='in_image')
    in_label = tf.compat.v1.placeholder(tf.int64, shape=(None, 1), name='in_label')
    input_graph_def = _parse_input_graph_proto(FLAGS.model_path)
    # We can remove redudant fixneurons if we want to get better performance but the accuracy may has 
    # a minor difference with the original quantized model.
    accuracy_mode = vitis_vai.wrap_conversion.AccuracyMode.ReserveReduantFixNeurons if FLAGS.mode == "normal" and model_config['model']['name'] == "mobilenet_v1_0.25_128" else \
        vitis_vai.wrap_conversion.AccuracyMode.Default
    vai_wego_graph = vitis_vai.create_wego_graph(
        input_graph_def=input_graph_def,
        accuracy_mode=accuracy_mode
    )

    wego_tf1_model='./resnet_v1_50_wego.pb'
    with tf.gfile.FastGFile(wego_tf1_model, mode='wb') as f:
        f.write(vai_wego_graph.SerializeToString())
    output_graph_def = tf.GraphDef()
    with open(wego_tf1_model, "rb") as f:
        output_graph_def.ParseFromString(f.read())

    sess.graph.as_default()
    _ = importer.import_graph_def(
        output_graph_def,
        name="",
        input_map={model_config['model']['input_node'] + ':0': in_image})
    logits = sess.graph.get_tensor_by_name(model_config['model']['output_node'] + ':0')
    if model_config['model']['name']=="efficientNet-edgetpu-S":
        softmax_res = tf.compat.v1.nn.softmax(logits)
        top5_prob,top5_catid = tf.compat.v1.nn.top_k(softmax_res,5)
    else:
        top5_prob,top5_catid = tf.compat.v1.nn.top_k(logits,5)
    input_img = read_image(FLAGS.img_url)

    # Create callable directly for better performance
    sess_callable = make_callable(sess, feed=[in_image],target=[top5_prob,top5_catid],fetch=[top5_prob,top5_catid,logits])
    if FLAGS.mode == "normal":
        print("[Info] running in normal mode...")
        run_normal(input_img)
    elif FLAGS.mode == 'perf':
        print("[Info] running in perf mode...")
        run_perf(input_img)  
    else:
        raise ValueError('unsupport running mode - %s, support list: [normal, perf]' % (FLAGS.mode))
    sess.close()
