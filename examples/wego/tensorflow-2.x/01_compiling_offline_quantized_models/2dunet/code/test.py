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
import sys
import random
import warnings
import threading

import tensorflow as tf 
import numpy as np
from PIL import Image
import argparse
import skimage.io as io
import time
from skimage.io import imread, imshow
from skimage.transform import resize
from tensorflow.compiler import vitis_vai

def get_arguments():
    """Parse all the arguments.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="TF2 Semantic Segmentation")

    parser.add_argument("--input_size", type=str, default='128,128', help="Input shape: [H, W]")
    #data config
    parser.add_argument("--img_path", type=str, default='./data/nuclei_data',
                        help="Path to the directory containing the cityscapes validation images.")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes to predict.")
    parser.add_argument("--thread", type=int, default=1,
                        help="Number of thread.")
    parser.add_argument("--eval_iter", type=int, default=6000, help="images number to run.")
    # model config
    parser.add_argument("--weight_file", type=str, default='quantized_model/quantized.h5',
                        help="Path to the final best weights.")
    parser.add_argument("--save_path", type=str, default='./results_visulization/',
                        help="where to save the vis results.")
    parser.add_argument("--return_seg", type=bool, default=True,
                        help="resturn gray prediction")
    parser.add_argument("--mode", type=str, default="normal",
                        help="select perf or normal.")

    return parser.parse_args()

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

def run_func():
    r = model(img_list_group[0])[0].numpy()
    pred=[r[i]]
    pred=np.array(pred)
    pred_val = (pred > 0.5).astype(np.uint8)
    pred_val = np.squeeze(pred_val)
    pred_val[pred_val==1] = 255
    #save prediction
    pred_val = Image.fromarray(pred_val)
    pred_val.save(output_path+'/'+n_list[0].replace('img','pred'))
    print("[Info] output result image: %s/%s" % (output_path, n_list[0].replace('img','pred')))

def run_thread(cnt):
    for j in range(cnt, batch_iter, nthreads):
        r = model(img_list_group[0])[0].numpy()
        for i in range(r.shape[0]):
            pred=[r[i]]
            pred=np.array(pred)
            pred_val = (pred > 0.5).astype(np.uint8)
            pred_val = np.squeeze(pred_val)
            pred_val[pred_val==1] = 255

if  __name__ == '__main__':
    args = get_arguments()

    # Set data parameters
    NUM_CLASS=args.num_classes
    BATCH = vitis_vai.get_target_info().batch
    nthreads = args.thread
    IMG_WIDTH, IMG_HEIGHT = map(int, args.input_size.split(','))
    IMG_CHANNELS = 3
    # set model parameters
    seed = 42
    
    # save prediction
    save_pred = args.return_seg
    output_path = args.save_path

    dir_path = ''
    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
    random.seed = seed
    np.random.seed = seed
    
    model = vitis_vai.create_wego_model(input_h5=args.weight_file)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    Img_path = args.img_path
    img_list_group=[]
    run_count = 0
    n_list = []
    batch_iter = args.eval_iter
    #for name in os.listdir(Img_path):
    name = os.listdir(Img_path)[0]
    #Read images
    img = imread(os.path.join(Img_path, name))[:,:,:IMG_CHANNELS]
    #Resize image to match training data
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    img /= 255.0
    n_list.append(name)
        
    img_tensor = np.zeros((BATCH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
    for i in range(BATCH):
        img_tensor[i] = np.array(img)
    x=tf.convert_to_tensor(img_tensor,dtype='float32') 
    img_list_group.append(x)

    r = model(img_list_group[0])[0].numpy()
    if args.mode == "normal":
        img_tensor = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
        img_tensor[0] = np.array(img)
        x=tf.convert_to_tensor(img_tensor,dtype='float32') 
        img_list_group.append(x)
        run_func()
    else:
        img_tensor = np.zeros((BATCH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
        for i in range(BATCH):
            img_tensor[i] = np.array(img)
        x=tf.convert_to_tensor(img_tensor,dtype='float32') 
        img_list_group.append(x)
        r_n = 20
        print("[INFO] start perf test..")
        print("[INFO] repeat running %d times with %d images...." %
              (r_n, batch_iter*BATCH))
        t = 0.0
        for i in range(r_n):
            t += do_run()
        print("=========== Perf Result ==============")
        print("Total Images: %d" % (batch_iter*BATCH * r_n))
        print('Use_time = [%0.2fs]' % (t))
        print('qps = [%0.2f]' % (float(batch_iter*BATCH) / (t / r_n)))




