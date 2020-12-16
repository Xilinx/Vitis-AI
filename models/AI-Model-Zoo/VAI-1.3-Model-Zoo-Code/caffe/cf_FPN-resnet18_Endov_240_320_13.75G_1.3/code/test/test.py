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


import argparse
import numpy as np
from PIL import Image
import os
import sys
import cv2
import time

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
# global parameters
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434))  # mean_values for B, G,R
IMG_SCALE = 1.0

INPUT_W, INPUT_H = 320, 240 # W, H 


def get_arguments():
    """Parse all the arguments.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Endov FPN_R18_OS16_LR Network")

    parser.add_argument("--caffepath", type=str, default=None, help="caffe_xilinx path.")
    #data config
    parser.add_argument("--imgpath", type=str, default='./data/Endov/val/',
                        help="Path to the directory of endov validation dataset.")                    
    parser.add_argument("--num-classes", type=int, default=3,
                        help="Number of classes to predict.")
    # model config
    parser.add_argument("--modelpath", type=str, default='./float/',
                        help="Path to the directory containing the deploy.prototxt and caffemodel.")
    parser.add_argument("--prototxt_file", type=str, default='Endov_FPN_R18_OS16_LR.prototxt',
                        help="the prototxt file for inference.")
    parser.add_argument("--weight_file", type=str, default='Endov_FPN_R18_OS16_LR.caffemodel',
                        help="Path to the final best caffemodel weights.")
    # others                    
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")                 
    parser.add_argument("--savepath", type=str, default='./results',
                        help="where to save the vis results.")
    parser.add_argument("--colorFormat", type=bool, default=False,
                        help="add corlors on results.")

    return parser.parse_args()

def label_img_to_color(img):
    label_to_color = {
        0: [0, 0, 0],
        1: [0, 255, 0],
        2: [0, 0, 255],
        3: [255, 0, 0]}

    img_height, img_width = img.shape
    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]
            img_color[row, col] = np.array(label_to_color[label])
    return img_color


def segment(net, img_file):
    im_ = cv2.imread(img_file)
    
    h, w = im_.shape[0], im_.shape[1]
    in_ = cv2.resize(im_, (INPUT_W, INPUT_H))
    in_ = in_ * 1.0
    in_ -= IMG_MEAN
    in_ *= IMG_SCALE
    # H x W x C --> C x H x W
    in_ = in_.transpose((2,0,1))
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)   # add batch dim
    net.blobs['data'].data[...] = in_
    #caffe
    net.forward()
    t_s = time.time()
    out = net.blobs['score'].data[0].argmax(axis=0)  
    gray_to_save = cv2.resize(out, dsize=(w,h), interpolation=cv2.INTER_NEAREST)
    pred_label_color = label_img_to_color(out)
    color_to_save = Image.fromarray(pred_label_color.astype(np.uint8))
    color_to_save = color_to_save.resize((w,h))

    return gray_to_save, color_to_save

    
def main():  
    args = get_arguments()
    add_path(os.path.join(args.caffepath, 'python/'))
    import caffe
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
    count = 0
    caffe.set_device(int(args.gpu))
    net = caffe.Net(os.path.join(args.modelpath, args.prototxt_file),os.path.join(args.modelpath, args.weight_file),caffe.TEST) 
    categories = os.listdir(args.imgpath)
    for c in categories:
        save_path = os.path.join(args.savepath, c)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for name in os.listdir(os.path.join(args.imgpath, c, 'Raw')):
            print(name, c)
            image_file = os.path.join(os.path.join(args.imgpath, c, 'Raw', name))
            raw_img = cv2.imread(image_file)
            assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
            gray_to_save, color_to_save = segment(net, image_file)
           
            cv2.imwrite(os.path.join(save_path, name[:-4] + '.png'), gray_to_save)
            if args.colorFormat:
                color_to_save.save(os.path.join(save_path, name[:-4] + '_color.png'))

if __name__ == '__main__':
    main()

