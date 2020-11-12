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

INPUT_W, INPUT_H = 320, 320 # W, H 
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255)]
classTypes =  ['BE', 'cancer', 'HGD' , 'polyp', 'suspicious']

def get_arguments():
    """Parse all the arguments.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Xilinx_FPN_8.9G Network")

    parser.add_argument("--caffepath", type=str, default=None, help="caffe_xilinx path.")

    #data config
    parser.add_argument("--imgpath", type=str, default='./data/EDD/images/',
                        help="Path to the directory containing the cityscapes validation dataset.")                    
    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of classes to predict.")

    # model config
    parser.add_argument("--modelpath", type=str, default='./float/',
                        help="Path to the directory containing the deploy.prototxt and caffemodel.")
    parser.add_argument("--prototxt_file", type=str, default='pytorch2caffe_mergebn2conv.prototxt',
                        help="the prototxt file for inference.")
    parser.add_argument("--weight_file", type=str, default='pytorch2caffe_mergebn2conv.caffemodel',
                        help="Path to the final best caffemodel weights.")
    # others                    
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")                 
    parser.add_argument("--savepath", type=str, default='./results',
                        help="where to save the vis results.")
    parser.add_argument("--colorFormat", type=bool, default=False,
                        help="add corlors on results.")

    return parser.parse_args()

def segment(net, img_file, name):
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
    pred_0 = net.blobs['score_0'].data[0].argmax(axis=0)  
    pred_1 = net.blobs['score_1'].data[0].argmax(axis=0)
    pred_2 = net.blobs['score_2'].data[0].argmax(axis=0)
    pred_3 = net.blobs['score_3'].data[0].argmax(axis=0)
    pred_4 = net.blobs['score_4'].data[0].argmax(axis=0)

    gray_0 = cv2.resize(pred_0, dsize=(w,h), interpolation=cv2.INTER_NEAREST)
    gray_1 = cv2.resize(pred_1, dsize=(w,h), interpolation=cv2.INTER_NEAREST)
    gray_2 = cv2.resize(pred_2, dsize=(w,h), interpolation=cv2.INTER_NEAREST)
    gray_3 = cv2.resize(pred_3, dsize=(w,h), interpolation=cv2.INTER_NEAREST)
    gray_4 = cv2.resize(pred_4, dsize=(w,h), interpolation=cv2.INTER_NEAREST)
    return gray_0, gray_1, gray_2, gray_3, gray_4
    
def main():  
    args = get_arguments()
    add_path(os.path.join(args.caffepath, 'python/'))
    import caffe
    img_path = args.imgpath
    save_path = args.savepath
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    count = 0
    caffe.set_device(int(args.gpu))
    net = caffe.Net(os.path.join(args.modelpath, args.prototxt_file),os.path.join(args.modelpath, args.weight_file),caffe.TEST) 
    img_lists = open('data/EDD/val_list.txt','r')
    objs = img_lists.readlines()
    N = len(objs)
    
    for item in objs:
        image_name = item.strip().split()[0]
        count += 1
        print('process image id: {}/{}'.format(count, N))
        image_file = os.path.join(img_path, image_name + '.jpg')
        raw_img = cv2.imread(image_file)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        gray_0, gray_1, gray_2, gray_3, gray_4 = segment(net, image_file, image_name)
        outputs = [gray_0, gray_1, gray_2, gray_3, gray_4]
        for i in range(len(outputs)):
            pred = outputs[i]    
            if not os.path.exists(os.path.join(save_path, str(classTypes[i]))):
                os.makedirs(os.path.join(save_path, str(classTypes[i])))
            cv2.imwrite(os.path.join(save_path, str(classTypes[i]), image_name + '.png'), pred)
        if args.colorFormat:
            for i in range(len(outputs)):
                y_true = cv2.imread(os.path.join(save_path, str(classTypes[i]), image_name + '.png'), 0)
                contours,_ = cv2.findContours(y_true, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if contours !=[]:
                    midVal = int(contours[0].shape[0]/2)
                    cv2.drawContours(raw_img, contours, -1, colors[i], 3)
                    cv2.putText(raw_img ,classTypes[i],(contours[0][midVal][0][0], contours[0][midVal][0][1]),  cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.imwrite(os.path.join(save_path, image_name + '_overlayer.png'), raw_img)

if __name__ == '__main__':
    main()

