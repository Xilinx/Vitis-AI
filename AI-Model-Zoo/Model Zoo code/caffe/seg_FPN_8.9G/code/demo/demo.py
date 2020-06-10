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

# PART OF THIS FILE AT ALL TIMES.


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
IMG_MEAN = np.array((104,117,123))  # mean_values for B, G,R
IMG_SCALE = 1.0

INPUT_W, INPUT_H = 512, 256 # W, H 
TARGET_W, TARGET_H = 2048, 1024

def get_arguments():
    """Parse all the arguments.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Xilinx_FPN_8.9G Network")

    parser.add_argument("--caffepath", type=str, help="caffe_xilinx path.")

    parser.add_argument("--num-classes", type=int, default=19,
                        help="Number of classes to predict.")
    # model config
    parser.add_argument("--modelpath", type=str, default='../../float/',
                        help="Path to the directory containing the deploy.prototxt and caffemodel.")
    parser.add_argument("--prototxt_file", type=str, default='test.prototxt',
                        help="the prototxt file for inference.")
    parser.add_argument("--weight_file", type=str, default='trainval.caffemodel',
                        help="Path to the final best caffemodel weights.")
    # others                    
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")                 
    parser.add_argument("--savepath", type=str, default='./results_visulization',
                        help="where to save the vis results.")
    parser.add_argument("--colorFormat", type=bool, default=True,
                        help="add corlors on results.")

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
        }

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]
            img_color[row, col] = np.array(label_to_color[label])
    return img_color

def segment(net, img_file, name):
    im_ = cv2.imread(img_file)
    w, h = TARGET_W, TARGET_H
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
    t = time.time() - t_s
    #print('it took {:.3f}s'.format(t))
    #save color_output
    gray_to_save = cv2.resize(out, dsize=(w,h), interpolation=cv2.INTER_NEAREST)
    pred_label_color = label_img_to_color(out)
    color_to_save = Image.fromarray(pred_label_color.astype(np.uint8))
    color_to_save = color_to_save.resize((w,h))
    return gray_to_save, color_to_save
    
def main():  
    args = get_arguments()
    add_path(args.caffepath)
    import caffe

    save_path = args.savepath
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    caffe.set_device(int(args.gpu))
    net = caffe.Net(os.path.join(args.modelpath, args.prototxt_file),os.path.join(args.modelpath, args.weight_file),caffe.TEST) 
    test_image = './demo/frankfurt_000000_000294_leftImg8bit.png'        
    image_name = os.path.basename(test_image)[:-4]
    assert os.path.exists(test_image), 'Path does not exist: {}'.format(test_image)
    gray_to_save, color_to_save = segment(net, test_image, image_name)

    cv2.imwrite(os.path.join(save_path, image_name + '.png'), gray_to_save)
    if args.colorFormat:
        color_to_save.save(os.path.join(save_path, image_name + '_color.png'))
        

if __name__ == '__main__':
    main()

