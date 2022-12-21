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
import numpy as np
import argparse
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import cv2
import os
from glob import glob
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.compiler import vitis_vai
import threading
import time
def get_arguments():
    """Parse all the arguments.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="TF2 Semantic Segmentation")

    parser.add_argument('--arch', default='cbr', choices=['cbr', 'crb'], \
                         help='model arch design: cbr=conv+bn+relu, crb=conv+relu+bn')
    parser.add_argument("--input_size", type=str, default='512,1024', help="Input shape: [H, W]")
    #data config
    parser.add_argument("--img_path", type=str, default='/scratch/data/cityscapes_20cls/val_images/',
                        help="Path to the directory containing the cityscapes validation images.")
    parser.add_argument("--num_classes", type=int, default=20,
                        help="Number of classes to predict.")
    parser.add_argument("--batch_iter", type=int, default=2000,
                        help="Number of batch to run.")
    parser.add_argument("--thread", type=int, default=1,
                        help="Number of thread.")
    # model config
    parser.add_argument("--weight_file", type=str, default='/workspace/quantized_erfnet.h5',
                        help="Path to the final best weights.")
    parser.add_argument("--save_path", type=str, default='./results_visulization_erfnet/512x1024/',
                        help="where to save the vis results.")
    parser.add_argument("--return_seg", type=bool, default=True,
                        help="resturn gray prediction")
    parser.add_argument("--add_color", type=bool, default=False,
                        help="merge corlors masks on the RGB images")
    parser.add_argument("--mode", type=str, default="normal",
                        help="select perf or normal.")

    return parser.parse_args()

def label_img_to_color(img):
    label_to_color = {
        0: [128, 64,128],
        1: [244, 35,232],
        2: [ 70, 70, 70],
        3: [102,102,156],
        4: [190,153,153],
        5: [153,153,153],
        6: [250,170, 30],
        7: [220,220,  0],
        8: [107,142, 35],
        9: [152,251,152],
        10: [ 70,130,180],
        11: [220, 20, 60],
        12: [255,  0,  0],
        13: [  0,  0,142],
        14: [  0,  0, 70],
        15: [  0, 60,100],
        16: [  0, 80,100],
        17: [  0,  0,230],
        18: [119, 11, 32],
        19: [0, 0, 0],
        }

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]
            img_color[row, col] = np.array(label_to_color[label])
    return img_color

def run_func():
    r=model(data)
    z = tf.squeeze(r[0][0])
    y = tf.math.argmax(z, axis=2)
    
    out = cv2.resize(y.numpy(), dsize=(W, H), interpolation=cv2.INTER_NEAREST)
    color_mask = label_img_to_color(out)
    color_mask = Image.fromarray(color_mask.astype(np.uint8)).convert('RGB')
    color_mask.save(os.path.join(args.save_path, 'color_'+ images_run[0]))
    print("[Info] output result image: %s/color_%s" % (args.save_path, images_run[0]))

def run_thread(cnt):
    for j in range(cnt, n_of_group_run, nthreads):
        r=model(data)
        z = tf.squeeze(r[0])
        y = tf.math.argmax(z, axis=2)
           

def do_run():
    threads = []

    for i in range(nthreads):
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
    args = get_arguments()

    h, w = map(int, args.input_size.split(','))
    num_classes = args.num_classes
    nthreads = args.thread
    model = vitis_vai.create_wego_model(input_h5=args.weight_file)       

    image_dir = args.img_path
    images = os.listdir(image_dir)
    images.sort()
    image_list = []
    images_run = []
    image_list_group = []
    batch = vitis_vai.get_target_info().batch
    
    image_file = images[0]
    image = load_img(os.path.join(image_dir, image_file))
    image = img_to_array(image)
    alpha = 0.5
    dims = image.shape
    H, W = image.shape[0], image.shape[1]
    if not os.path.exists(args.save_path):
       os.makedirs(args.save_path)
        
    raw_img = image.copy()
    h, w = map(int, args.input_size.split(','))
    image = cv2.resize(image, (w, h))
    x = image.copy()
    x = preprocess_input(np.expand_dims(x, axis=0))
    x = x / 255.0
    x=x[0]
    images_run.append(image_file)
    if args.mode == "normal":
        run_size = 1
    else:
        run_size = batch
    img_tensor = np.zeros((run_size, h, w, 3), dtype=np.float32)
    for i_l in range(run_size):
        img_tensor[i_l] = np.array(x)
    data=tf.convert_to_tensor(img_tensor,dtype='float32')
            
    n_of_group_run = args.batch_iter
    
    r=model(data)
    if args.mode == "normal":
        run_func()
    else:
        r_n = 2
        print("[INFO] start perf test..")
        print("[INFO] repeat running %d times with %d images...." %
              (r_n, n_of_group_run*batch))
        t = 0.0
        for i in range(r_n):
            t += do_run()
        print("=========== Perf Result ==============")
        print("Total Images: %d" % (n_of_group_run*batch * r_n))
        print('Use_time = [%0.2fs]' % (t))
        print('qps = [%0.2f]' % (float(n_of_group_run*batch) / (t / r_n)))
