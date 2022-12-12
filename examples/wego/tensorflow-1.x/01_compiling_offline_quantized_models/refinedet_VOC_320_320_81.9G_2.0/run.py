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
# bash

import os
import cv2
import time
import threading
import numpy as np
import tensorflow as tf

from argparse import ArgumentParser
from utils_tf import pboxes_vgg_voc, Encoder
from tensorflow.contrib import vitis_vai
from tqdm import tqdm

def parse_args():
    parser = ArgumentParser(description="Refinedet vgg evaluation on VOC")
    parser.add_argument('--model_dir', '-m', type=str,
            default='./model/quantized_refinedet_voc.pb',
            help='path to frozen graph')
    parser.add_argument('--img_url', default='', help='source input image.')
    parser.add_argument('--threads', '-tm', type=int,
            default=6,
            help='the thread number for running evaluation')
    parser.add_argument('--eval_iter', '-i', type=int,
            default=960,
            help='eval iterations.')
    parser.add_argument('--mode', '-md', type=str,
            default='normal',
            help='normal or perf mode')
    parser.add_argument('--score-threshold','-t', type=float,
            default=0.005, help='score threshold')
    return parser.parse_args()


run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
run_metadata = tf.compat.v1.RunMetadata()

VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

def get_map_dict():
    labels_to_names = {}
    for name, pair in VOC_LABELS.items():
        labels_to_names[pair[0]] = name
    return labels_to_names

def preprocess(image):
    image = np.array(cv2.resize(image, (320,320)),dtype=np.float32)
    R_MEAN = 123.68
    G_MEAN = 116.78
    B_MEAN = 103.94
    mean = np.array([B_MEAN, G_MEAN, R_MEAN], dtype=np.float32)
    std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    image = (image - mean) / std
    return image
    
def read_image(image_path):
    img = cv2.imread(image_path)
    img_shape = img.shape[:2]
    input_img = preprocess(img)
    return np.expand_dims(input_img, 0), img, img_shape

def postprocess(output_dict,img,img_shape):
    arm_cls = output_dict['arm_cls']
    arm_loc = output_dict['arm_loc']
    odm_cls = output_dict['odm_cls']
    odm_loc = output_dict['odm_loc']
    loc, label, prob = encoder.decode_batch(arm_cls, arm_loc, odm_cls, odm_loc, 0.45, 200, device=0)[0]
    valid_mask = np.logical_and((loc[:, 2] - loc[:, 0] > 0), (loc[:, 3] - loc[:, 1] > 0))
    for i in range(prob.shape[0]-1, -1,-1):
        if not valid_mask[i]:
            continue
        score = prob[i]
        if score < args.score_threshold:
            break
        xmin = int(loc[i][0] * img_shape[1])
        ymin = int(loc[i][1] * img_shape[0])
        xmax = int(loc[i][2] * img_shape[1])
        ymax = int(loc[i][3] * img_shape[0])
        class_name = map_dict[label[i]]
        
        fs = 0.8
        tf = 1
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax), colors_tableau[label[i]],tf)
        w, h = cv2.getTextSize(class_name, 0, fontScale=fs, thickness=tf)[0]  # text width, height
        outside = ymin - h - 3 >= 0  # label fits outside box
        cv2.putText(img,
                    class_name, (xmin, ymin - 2 if outside else ymin + h + 2),
                    0,
                    fs,
                    colors_tableau[label[i]],
                    thickness=tf,
                    lineType=cv2.LINE_AA)

    cv2.imwrite("result.png",img)
    print("[Info] result image path: result.png")

def run_thread(images, t_id, n_threads):
    begin, step = t_id, n_threads
    for j in range(begin, len(images), step):
        output_dict = sess.run(tensor_dict, feed_dict={image_tensor: images[j]}) 
        
def run(images, n_threads):
    thread_list = []
    for t_id in range(0, n_threads):
        t = threading.Thread(
            target = run_thread,
            args = (images, t_id, n_threads)
        )
        thread_list.append(t)
    
    st = time.perf_counter()
    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join()
    et = time.perf_counter()
    
    return (et-st)

def run_perf(args, img):
    # create batch images based on batch size;
    batch_size = vitis_vai.get_target_info().batch
    batch_images = np.concatenate([img for i in range(batch_size)], 0)
    # run images with repeated batch images;
    repeat_batch = int(args.eval_iter/batch_size)
    all_images = [batch_images] * repeat_batch

    n_images = repeat_batch * len(batch_images)

    print("[Info] warm up ...")
    for i in tqdm(range(5)):
      run(all_images, args.threads)

    r_n = 20
    print("[Info] begin to run inference using %d images with %d times." % (n_images, r_n))
 
    t = 0.0
    for i in tqdm(range(r_n)):
      t_ = run(all_images, args.threads)
      t += t_
    print("===================== Perf Result =====================")
    print("[Total Images] %d" % (r_n * n_images))
    print("[Total Time]   %0.6fs" % float(t))
    print("[FPS]          %0.2f" % (float(n_images) / (t / r_n)))


def run_normal(args,input_img,img,img_shape):
    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: input_img}) 
    postprocess(output_dict,img,img_shape)



if __name__=='__main__':
    args = parse_args()
    mode = args.mode
    sess = tf.compat.v1.Session()
    with tf.io.gfile.GFile(args.model_dir, 'rb') as fid:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(fid.read()) 
        vai_wego_graph = vitis_vai.create_wego_graph(
                input_graph_def=graph_def)

        tf.import_graph_def(vai_wego_graph, name='')
    image_tensor = sess.graph.get_tensor_by_name('image:0')
    output_keys = ['arm_cls', 'arm_loc', 'odm_cls', 'odm_loc']
    tensor_dict = {}
    for key in output_keys:
        tensor_dict[key] = sess.graph.get_tensor_by_name(key + ":0") 

    # for postprocessing
    pboxes = pboxes_vgg_voc()
    encoder = Encoder(pboxes)
    map_dict = get_map_dict()

    input_img, img, img_shape = read_image(args.img_url)
    if args.mode == "normal":
        print("[Info] running in normal mode...")
        run_normal(args,input_img,img,img_shape)
    elif args.mode == 'perf':
        print("[Info] running in perf mode...")
        run_perf(args, input_img)  
    else:
        raise ValueError('unsupport running mode - %s, support list: [normal, perf]' % (args.mode))
