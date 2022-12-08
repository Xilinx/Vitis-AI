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
import os.path
import time
import threading
import argparse
import numpy as np
import torch
import cv2
import torch.utils.data as udata
import wego_torch

from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm
import utils

# parse user arguments
parser = argparse.ArgumentParser(description='PyTorch DRUnet model')  
parser.add_argument('--img_url', default='', help='source input image.')
parser.add_argument('--model_dir', type=str, default='./float', help='path to model dir')
parser.add_argument('--threads', help='The threads number for running the applications', type=int,default=6)
parser.add_argument('--mode', default='normal', help="running mode.")

args = parser.parse_args()

def read_image(image_path):
    img = cv2.imread(image_path)
    # convert to 4 dimension torch tensor
    img = utils.uint2tensor4(img)
    return img

def run_thread(model, images, t_id, n_threads):
    begin, step = t_id, n_threads
    for j in range(begin, len(images), step):
        denoise_imgs = model(images[j])
        
def run(model, images, n_threads):
    thread_list = []
    for t_id in range(0, n_threads):
        t = threading.Thread(
            target = run_thread,
            args = (model, images, t_id, n_threads)
        )
        thread_list.append(t)
    
    st = time.perf_counter()
    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join()
    et = time.perf_counter()
    
    return (et-st)

def run_normal(args, img, wego_mod):
    with torch.no_grad():
        denoise_img = wego_mod(img)
        data_saved = utils.tensor2uint(denoise_img)
        data_saved = data_saved[:, :, [2, 1, 0]]
        name = "result.png"
        print("[Info] output result denoise image: %s" % (name))
        utils.imsave(data_saved, name)

def run_perf(args, img, batch, wego_mod):
    # create batch images based on batch size;
    batch_images = torch.cat([img for i in range(batch)], 0)
    # run images with repeated batch images;
    repeat_batch = 20
    all_images = [batch_images] * repeat_batch

    n_images = repeat_batch * len(batch_images)
    r_n = 20
    print("[Info] begin to run inference using %d images with %d times." % (n_images, r_n))
 
    t = 0.0
    for i in tqdm(range(r_n)):
      t_ = run(wego_mod, all_images, args.threads)
      t += t_
    print("===================== Perf Result =====================")
    print("[Total Images] %d" % (r_n * n_images))
    print("[Total Time]   %0.6fs" % float(t))
    print("[FPS]          %0.2f" % (float(n_images) / (t / r_n)))

def run_normal(args, img, wego_mod):
    with torch.no_grad():
        denoise_img = wego_mod(img)
        data_saved = utils.tensor2uint(denoise_img)
        data_saved = data_saved[:, :, [2, 1, 0]]
        name = "result.png"
        print("[Info] output result denoise image: %s" % (name))
        utils.imsave(data_saved, name)

def main():
    # loading original quantized torchscript module
    model_path = args.model_dir
    mod = torch.jit.load(model_path)

    # Create wego module 
    wego_mod = wego_torch.compile(mod, wego_torch.CompileOptions(
        inputs_meta = [
        wego_torch.InputMeta(torch.float, [1, 3, 528, 608])
        ]
    ))

    target_info = wego_torch.get_target_info()
    print("[Info] target_info: %s" % str(target_info))
    batch = target_info.batch
     
    # read one single image
    img = read_image(args.img_url)

    if args.mode == 'normal':
        print("[Info] running in normal mode...")
        run_normal(args, img, wego_mod)
    elif args.mode == 'perf':
        print("[Info] running in perf mode...")
        run_perf(args, img, batch, wego_mod)  
    else:
        raise ValueError('unsupport running mode - %s, support list: [normal, perf]' % (args.mode))

if __name__ == '__main__':
    main()
