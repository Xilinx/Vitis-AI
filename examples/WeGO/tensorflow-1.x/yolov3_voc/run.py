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
import cv2
import time
import threading
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.contrib import vitis_vai
from yolov3_predictor import yolo_predictor
from config import Yolov3VOCConfig as Config

tf.app.flags.DEFINE_string('model_dir', './model/quantized_yolov3_voc.pb',
                           'TensorFlow \'GraphDef\' file to load.')
tf.app.flags.DEFINE_string('img_url', '',
                           'source input image')
tf.app.flags.DEFINE_string('input_node', 'input_1', 'input node of pb model')
tf.app.flags.DEFINE_string(
    'output_node', 'conv2d_59/BiasAdd,conv2d_67/BiasAdd,conv2d_75/BiasAdd',
    'ouput node of pb model')
tf.app.flags.DEFINE_integer('input_height', 416, 'input height of pb model')
tf.app.flags.DEFINE_integer('input_width', 416, 'input width of pb model')
tf.app.flags.DEFINE_integer('threads', 8,
                            'The thread number for running evaluation.')
tf.app.flags.DEFINE_integer('eval_iter', 960,
                            'eval iterations.')
tf.app.flags.DEFINE_string('mode', '', 'normal or perf mode')

FLAGS = tf.app.flags.FLAGS

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    ih, iw, _ = image.shape
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    new_image = np.ones((h, w, 3), np.uint8) * 128
    h_start = (h-nh)//2
    w_start = (w-nw)//2
    new_image[h_start:h_start+nh, w_start:w_start+nw, :] = image
    return new_image

def preprocess(image, input_size):
    # BGR -> RGB
    image = image[..., ::-1]
    if input_size != (None, None):
        assert input_size[0] % 32 == 0, 'Multiples of 32 required'
        assert input_size[1] % 32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(input_size)))
    else:
        image_h, image_w, _ = image.shape
        new_image_size = (image_w - (image_w % 32), image_h - (image_h % 32))
        boxed_image = letterbox_image(image, new_image_size)

    boxed_image = np.array(boxed_image, dtype='float32')
    return boxed_image/255.

def read_image(image_path):
    img = cv2.imread(image_path)
    img_shape = img.shape[:2]
    input_img = preprocess(img,(FLAGS.input_width,FLAGS.input_height))
    return np.expand_dims(input_img, 0), img, img_shape

def postprocess(img,img_shape,nms_results):
    nms_boxes, nms_scores, nms_classes, valid_num = nms_results
    out_boxes = nms_boxes[:valid_num]    # [valid_num, 4]
    out_class = nms_classes[:valid_num]  # [valid_num] 
    out_scores = nms_scores[:valid_num]  # [valid_num]
    for i, c in reversed(list(enumerate(out_class))):
        class_name = class_names[int(c)]
        box = out_boxes[i]
        score = out_scores[i]

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(img_shape[0], np.floor(
            bottom + 0.5).astype('int32'))
        right = min(img_shape[1], np.floor(
            right + 0.5).astype('int32'))
        
        fs = 0.8
        tf = 1
        cv2.rectangle(img,(left,top),(right,bottom), colors[int(c)],1)
        w, h = cv2.getTextSize(class_name, 0, fontScale=fs, thickness=tf)[0]  # text width, height
        outside = top - h - 3 >= 0  # label fits outside box
        cv2.putText(img,
                    class_name, (left, top - 2 if outside else top + h + 2),
                    0,
                    fs,
                    colors[int(c)],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
    cv2.imwrite("result.png",img)
    print("[Info] result image path: result.png")

def run_thread(images,image_shapes, t_id, n_threads):
    begin, step = t_id, n_threads
    for j in range(begin, len(images), step):
        nms_boxes, nms_scores, nms_classes, valid_detections = sess.run(
        [pred_boxes, pred_scores, pred_classes, pred_detections],
        feed_dict={input_x: images[j], input_image_shape: image_shapes[j]})

def run(images,image_shapes, n_threads):
    thread_list = []
    for t_id in range(0, n_threads):
        t = threading.Thread(
            target = run_thread,
            args = (images,image_shapes, t_id, n_threads)
        )
        thread_list.append(t)
    
    st = time.perf_counter()
    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join()
    et = time.perf_counter()
    
    return (et-st)

def run_perf(img,img_shape):
    # create batch images based on batch size;
    batch = vitis_vai.get_target_info().batch
    batch_images = np.concatenate([img for i in range(batch)], 0)
    batched_img_shapes = [img_shape] * batch
    
    # run images with repeated batch images;
    repeat_batch = int(FLAGS.eval_iter/batch)
    all_images = [batch_images] * repeat_batch
    all_img_shapes = [batched_img_shapes] * repeat_batch
    n_images = repeat_batch * len(batch_images)
    print("[Info] warm up ...")
    for i in tqdm(range(5)):
      run(all_images,all_img_shapes, FLAGS.threads)
    r_n = 20
    print("[Info] begin to run inference using %d images with %d times." % (n_images, r_n))
 
    t = 0.0
    for i in tqdm(range(r_n)):
      t_ = run(all_images,all_img_shapes, FLAGS.threads)
      t += t_
    print("===================== Perf Result =====================")
    print("[Total Images] %d" % (r_n * n_images))
    print("[Total Time]   %0.6fs" % float(t))
    print("[FPS]          %0.2f" % (float(n_images) / (t / r_n)))

def run_normal(input_img,img,img_shape):
    nms_boxes, nms_scores, nms_classes, valid_detections = sess.run(
        [pred_boxes, pred_scores, pred_classes, pred_detections],
        feed_dict={input_x: input_img, input_image_shape: [img_shape]})
    postprocess(img,img_shape,(nms_boxes[0],nms_scores[0],nms_classes[0],valid_detections[0]))

if __name__ == "__main__":

    config = Config()
    class_names = config.classes
    colors = config.colors
    predictor = yolo_predictor(config)

    sess = tf.compat.v1.Session()
    with tf.io.gfile.GFile(FLAGS.model_dir, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read()) 
        graph_def = vitis_vai.create_wego_graph(
            input_graph_def=graph_def,
            feed_dict={"input_1": [-1, FLAGS.input_height, FLAGS.input_width, 3]}
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

    input_img, img, img_shape = read_image(FLAGS.img_url)


    if FLAGS.mode == "normal":
        print("[Info] running in normal mode...")
        run_normal(input_img,img,img_shape)

    elif FLAGS.mode == "perf":
        print("[Info] running in perf mode...")
        run_perf(input_img,img_shape)  
    else:
        raise ValueError('unsupport running mode - %s, support list: [normal, perf]' % (FLAGS.mode))

    sess.close()
