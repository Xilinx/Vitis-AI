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
import torch
import wego_torch
import torchvision.transforms as Transforms
import time
import numpy as np
import argparse
from PIL import Image
import math
import threading
from tqdm import tqdm
import cv2

parser = argparse.ArgumentParser("face_quality")
parser.add_argument('--img_url', default='', help='source input image.')
parser.add_argument('--threads', help='The threads number for running the applications', type=int, default=6)
parser.add_argument('--size', nargs='+', type=int, default=[80, 60], help='input size')
parser.add_argument('--model_path', default='', type=str, metavar='PATH',help='path to pretrained (default: none)')
parser.add_argument('--mean', type=float, default=[0.5, 0.5, 0.5])
parser.add_argument('--std', type=float, default=[0.5, 0.5, 0.5])
parser.add_argument('--mode', default='normal', help="running mode.")


args = parser.parse_args()

def load_image(args):
    transform = Transforms.Compose([Transforms.Resize(size = args.size),
                                    Transforms.Grayscale(3), # gray
                                    Transforms.ToTensor(),
                                    Transforms.Normalize(mean = args.mean, std = args.std),
                                ])

    img_path = args.img_url
    ori_img = cv2.imread(img_path)
    image = Image.open(img_path)
    image = transform(image)
    image = torch.unsqueeze(image, 0)
    
    return (ori_img, image)

def run_thread(model, images, t_id, n_threads):
    begin, step = t_id, n_threads

    with torch.no_grad():
        for j in range(begin, len(images), step):
            img = images[j]
            points_output, quality_output = model(img)

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

def run_perf(args, img, batch, wego_mod):
    # create batch images based on batch size;
    batch_images = torch.cat([img for i in range(batch)], 0)
    # run images with repeated batch images;
    repeat_batch = 400
    all_images = [batch_images] * repeat_batch

    n_images = repeat_batch * len(batch_images)
    print("[Info] begin to run inference with %d images." % (n_images))

    r_n = 20
    t = 0.0
    for i in tqdm(range(r_n)):
      t_ = run(wego_mod, all_images, args.threads)
      t += t_
    print("===================== Perf Result =====================")
    print("[Total Images] %d" % (r_n * n_images))
    print("[Total Time]   %0.6fs" % float(t))
    print("[FPS]          %0.2f" % (float(n_images) / (t / r_n)))

def run_normal(args, ori_img, img, wego_mod):
    with torch.no_grad():
        points_output, quality_output = wego_mod(img)
        points = points_output[0].cpu().data.numpy().reshape((2, 5)).T
        # compute points for the original image
        points[:,0] = points[:,0] * ori_img.shape[1] / 60.0
        points[:,1] = points[:,1]*ori_img.shape[0] / 80.0

        q = quality_output[0].data.cpu().numpy()
        q = 1/(1.+math.e**-((3.*q-600.)/150.))
        # draw points in original image
        for p in points:
            cv2.circle(ori_img,(int(p[0]),int(p[1])),2,(55,255,155),1)
        # write result
        name = "result.jpg"
        print("[Info] output result image: %s" % (name))
        cv2.imwrite(name, ori_img)
          
if __name__ == '__main__':

    # loading original quantized torchscript module
    model_path = args.model_path
    mod = torch.jit.load(model_path)

    # Create wego module 
    wego_mod = wego_torch.compile(mod, wego_torch.CompileOptions(
        inputs_meta = [
        wego_torch.InputMeta(torch.float, [1, 3, 80, 60])
        ]
    ))

    # set model as eval mode
    wego_mod.eval()

      # get wego target info
    target_info = wego_torch.get_target_info()
    print("[Info] target_info: %s" % str(target_info))
    batch = target_info.batch

    ori_img, img = load_image(args)

    if args.mode == 'normal':
        print("[Info] running in normal mode...")
        run_normal(args, ori_img, img, wego_mod)
    elif args.mode == 'perf':
        print("[Info] running in perf mode...")
        run_perf(args, img, batch, wego_mod)  
    else:
        raise ValueError('unsupport running mode - %s, support list: [normal, perf]' % (args.mode))
