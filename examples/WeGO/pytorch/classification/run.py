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
import threading
import argparse
import time
import random
import torch
import wego_torch
import torchvision
import validators
import requests
from PIL import Image
from typing import Tuple

import torchvision.transforms as transforms

from tqdm import tqdm
from config import load_config

parser = argparse.ArgumentParser()

parser.add_argument('--img_url', default='', help='source input image.')
parser.add_argument('--root_model_path', default='.', help='root path of models.')
parser.add_argument('--threads', help='The threads number for running the applications', type=int, default=6)
parser.add_argument('--config_file', default='', help='model config file.')
parser.add_argument('--mode', default='normal', help="running mode.")

args, _ = parser.parse_known_args()

model_config = load_config(args.config_file)

def get_image_from_url(model_config, url=''):
    mean = model_config['preprocess']['mean']
    std = model_config['preprocess']['std']
    size = model_config['preprocess']['input_size']
    resize = model_config['preprocess']['resize']
    print("[Info] preprocess mean - ", mean, ", std - ", std)

    normalize = transforms.Normalize(mean=mean, std=std)

    img_transforms = transforms.Compose([
          transforms.Resize(resize),
          transforms.CenterCrop(size),
          transforms.ToTensor(),
          normalize,
    ])
    
    # download the image from web if uri is valid.
    if (validators.url(url)):
        img = Image.open(requests.get(url, stream=True).raw)
    else:
        img = Image.open(url)

    img = img_transforms(img)
    return img

def get_categories():
    # https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories

def cal_topk(output, topk):
    topk_prob, topk_catid = torch.topk(output, topk)
    return topk_prob, topk_catid

def run_thread(model, images, t_id, n_threads):    
    begin, step = t_id, n_threads  
    with torch.no_grad():
        for j in range(begin, len(images), step):
            image = images[j]
            # compute output
            output = model(image)
            # compute top5
            prob = torch.nn.functional.softmax(output, dim=1)
            top5_prob, top5_catid = cal_topk(prob, 5)

def run(model, all_images_labels, n_threads):
    thread_list = []
  
    for t_id in range(0, n_threads):
        t = threading.Thread(
            target = run_thread,
            args = (model, all_images_labels, t_id, n_threads)
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
    batch_images = torch.cat([torch.unsqueeze(img, 0) for i in range(batch)], 0)
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

def run_normal(args, img, wego_mod):
    with torch.no_grad():
        img = torch.unsqueeze(img, 0)
        output = wego_mod(img)
        prob = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = cal_topk(prob, 5)
        categories = get_categories()
        print("=====================TopK Result=====================")
        for i in range(top5_prob.size(0)):
            print(categories[top5_catid[i]], top5_prob[i].item())

def main():
    print("running model:", model_config['model']['name'])
 
    # load original quantized torchscript module
    rpath = model_config['model']['rpath']
    model_path = os.path.join(args.root_model_path, rpath)
    mod = torch.jit.load(model_path)

    input_shape = model_config['model']['input_shape']

    # We can remove redudant fixneurons if we want to get better performance but the accuracy may has 
    # a minor difference with the original quantized model.
    accuracy_mode = wego_torch.AccuracyMode.ReserveReduantFixNeurons if args.mode == "normal" else \
        wego_torch.AccuracyMode.Default
    # Create wego module by compiling the original qunatized torchscript model
    wego_mod = wego_torch.compile(mod, wego_torch.CompileOptions(
        inputs_meta = [
        wego_torch.InputMeta(torch.float, [1] + input_shape)
        ],
        accuracy_mode=accuracy_mode
    ))
  
    target_info = wego_torch.get_target_info()
    print("[Info] target_info: %s" % str(target_info))
    batch = target_info.batch
    
    # read image
    img = get_image_from_url(model_config, args.img_url)

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
