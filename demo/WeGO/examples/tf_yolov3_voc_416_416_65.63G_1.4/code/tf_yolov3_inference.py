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
import math
import time
import threading
import itertools
import collections
import numpy as np
from input_fn import calib_input
from tqdm import tqdm
import tensorflow as tf
from tensorflow.contrib import vitis_vai
from tensorflow.python.client import timeline
from yolov3_predictor import yolo_predictor
from config import Yolov3VOCConfig as Config

tf.app.flags.DEFINE_string('input_graph', '../float_model/yolov3_voc.pb',
                           'TensorFlow \'GraphDef\' file to load.')
tf.app.flags.DEFINE_string('eval_image_path', '',
                           'The directory where put the eval images')
tf.app.flags.DEFINE_string('eval_image_list', '',
                           'file has validation images list')
tf.app.flags.DEFINE_string('input_node', 'input_1', 'input node of pb model')
tf.app.flags.DEFINE_string(
    'output_node', 'conv2d_59/BiasAdd,conv2d_67/BiasAdd,conv2d_75/BiasAdd',
    'ouput node of pb model')
tf.app.flags.DEFINE_integer('input_height', 416, 'input height of pb model')
tf.app.flags.DEFINE_integer('input_width', 416, 'input width of pb model')
tf.app.flags.DEFINE_string('result_file', 'voc2007_pred_results.txt',
                           'The directory of output results')
tf.app.flags.DEFINE_boolean('use_quantize', False, 'quantize or not')
tf.app.flags.DEFINE_integer('nthreads', '12',
                            'The thread number for running evaluation.')
tf.app.flags.DEFINE_integer('batch', '8',
                            'The batch size for running evaluation.')
tf.app.flags.DEFINE_string('mode', '', 'accuracy or perf mode')

FLAGS = tf.app.flags.FLAGS


def write_items_to_file(image_id, items, fw):
    for item in items:
        fw.write(image_id + " " +
                 " ".join([str(comp) for comp in item]) + "\n")


def write_detection_result(detection_result, fw):
    for img_id, items in detection_result.items():
        write_items_to_file(img_id, items, fw)


def detection_thread(images, image_ids, input_image_shapes, class_names, t_id, detection_result):
    batch, threads = FLAGS.batch, FLAGS.nthreads
    begin = t_id * batch
    step = threads * batch
    image_num = len(images)
    for j in range(begin, image_num, step):
        run_size = batch
        if((image_num - j) < batch):
            run_size = image_num - j
        begin, end = j, j + run_size
        input_images, img_ids, input_shapes = images[begin:
                                                     end], image_ids[begin: end], input_image_shapes[begin: end]

        nms_boxes, nms_scores, nms_classes, valid_detections = sess.run(
            [pred_boxes, pred_scores, pred_classes, pred_detections],
            feed_dict={input_x: input_images, input_image_shape: input_shapes})

        if FLAGS.mode == "perf":
            continue

        for b, img_id, input_shape in zip(range(0, run_size), img_ids, input_shapes):
            valid_num = valid_detections[b]  # 1
            out_boxes = nms_boxes[b]  # [max_detections, 4]
            out_boxes = out_boxes[:valid_num]  # [valid_num, 4]
            out_class = nms_classes[b]  # [max_detections]
            out_class = out_class[:valid_num]  # [valid_num]
            out_scores = nms_scores[b]
            out_scores = out_scores[:valid_num]

            items = []
            for i, c in reversed(list(enumerate(out_class))):
                predicted_class = class_names[int(c)]
                box = out_boxes[i]
                score = out_scores[i]

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(input_shape[0], np.floor(
                    bottom + 0.5).astype('int32'))
                right = min(input_shape[1], np.floor(
                    right + 0.5).astype('int32'))
                item = [predicted_class, score, left, top, right, bottom]
                items.append(item)
            detection_result[img_id] = items


def run_detection_batch(images, img_ids, input_shapes, class_names):
    thread_list = []
    detection_result = {}
    for t_id in range(0, FLAGS.nthreads):
        t = threading.Thread(
            target=detection_thread,
            args=(images, img_ids, input_shapes, class_names, t_id, detection_result))
        thread_list.append(t)

    st = time.perf_counter()
    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join()
    et = time.perf_counter()

    return detection_result, (et-st)


if __name__ == "__main__":

    config = Config()
    class_names = config.classes
    predictor = yolo_predictor(config)

    sess = tf.compat.v1.Session()
    with tf.io.gfile.GFile(FLAGS.input_graph, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read()) 
        # Create vitis-ai wego graph for cloud DPU acceleration
        graph_def = vitis_vai.create_wego_graph(
            target="DPUCVDX8H_ISA1_F2W2_8PE",
            input_graph_def=graph_def,
            feed_dict={
                "input_1": [-1, FLAGS.input_height, FLAGS.input_width, 3]}
        )
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    sess.run(tf.compat.v1.global_variables_initializer())

    input_x = sess.graph.get_tensor_by_name(FLAGS.input_node + ':0')
    output_y = []
    for node in FLAGS.output_node.split(","):
        output_y.append(sess.graph.get_tensor_by_name(node + ":0"))

    input_image_shape = tf.compat.v1.placeholder(tf.int32, shape=(None, 2))
    pred_boxes, pred_scores, pred_classes, pred_detections = predictor.predict(
        output_y, input_image_shape)

    mode = FLAGS.mode
    if mode != "accuracy" and mode != "perf":
        raise ValueError(
            "Unsupported mode, support values: [ %s, %s]." %
            ("accuracy", "perf"))

    load_size = None if mode == "accuracy" else 960

    print("[INFO] loading images...")
    img_list, img_ids, img_input_shapes = calib_input(
        FLAGS.eval_image_path,
        FLAGS.eval_image_list,
        FLAGS.input_height,
        FLAGS.input_width,
        load_size)

    if mode == "perf":
        r_n = 20
        print("[INFO] start running perf...")
        print("[INFO] repeat running %d times with %d images." %
              (r_n, load_size))
        t = 0.0
        for i in tqdm(range(r_n)):
            _, t_ = run_detection_batch(
                img_list, img_ids, img_input_shapes, class_names)
            t += t_
        print("===================== Perf Result =====================")
        print("[Total Images] %d" % (r_n * load_size))
        print("[Total Time]   %0.6fs" % float(t))
        print("[FPS]          %0.2f" % (float(load_size) / (t / r_n)))
    else:
        fw = open(FLAGS.result_file, "w")
        print("[INFO] start running detection...")
        detection_result, d_time = run_detection_batch(
            img_list, img_ids, img_input_shapes, class_names)
        print("[INFO] writing detection result to file: ", FLAGS.result_file)
        write_detection_result(detection_result, fw)
        print("[INFO] evaluating detection results...")
        print("===================== Accuracy Result =====================")
        fw.close()

    sess.close()
