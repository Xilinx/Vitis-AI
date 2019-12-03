#!/usr/bin/env python
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

import sys
import cv2
import numpy as np
import os,sys,timeit,json
from os import listdir
from os.path import isfile, join
import scipy.misc
import logging as log
import argparse

from vai.dpuv1.rt import xdnn, xdnn_io
from vai.dpuv1.utils.postproc import yolo
from yolo_utils import bias_selector, saveDetectionDarknetStyle, yolo_parser_args
from yolo_utils import draw_boxes, generate_colors
from get_mAP_darknet import calc_detector_mAP

def extant_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if x == "-":
      # skip file check and allow empty string
      return ""

    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x

def prep_image(image_file, net_width, net_height, pix_scale, pad_val, img_transpose, ch_swp):
    img = cv2.imread(image_file)
    orig_shape = img.shape
    height, width, __ = img.shape
    newdim = max(height, width)
    scalew = float(width)  / newdim
    scaleh = float(height) / newdim
    maxdim = max(net_width, net_height)
    neww   = int(maxdim * scalew)
    newh   = int(maxdim * scaleh)
    img    = cv2.resize(img, (neww, newh))

    if img.dtype != np.float32:
        img = img.astype(np.float32, order='C')

    img = img * pix_scale

    height, width, channels = img.shape
    newdim = max(height, width)
    letter_image = np.zeros((newdim, newdim, channels))
    letter_image[:, :, :] = pad_val
    if newdim == width:
        letter_image[(newdim-height)/2:((newdim-height)/2+height),0:width] = img
    else:
        letter_image[0:height,(newdim-width)/2:((newdim-width)/2+width)] = img

    img = letter_image

    img = np.transpose(img, (img_transpose[0], img_transpose[1], img_transpose[2]))

    ch = 3*[None]
    ch[0] = img[0,:,:]
    ch[1] = img[1,:,:]
    ch[2] = img[2,:,:]
    img   = np.stack((ch[ch_swp[0]],ch[ch_swp[1]],ch[ch_swp[2]]))

    return img, orig_shape


def yolo_gpu_inference(backend_path,
                       image_dir,
                       deploy_model,
                       weights,
                       out_labels,
                       IOU_threshold,
                       scorethresh,
                       mean_value,
                       pxscale,
                       transpose,
                       channel_swap,
                       yolo_model,
                       num_classes, args):

    # Setup the environment
    images = xdnn_io.getFilePaths(args['images'])
    if(args['golden'] or args['visualize']):
      assert args['labels'], "Provide --labels to compute mAP."
      assert args['results_dir'], "For accuracy measurements, provide --results_dir to save the detections."
      labels = xdnn_io.get_labels(args['labels'])
      colors = generate_colors(len(labels))

    # Select postproc and biases
    if   args['yolo_version'] == 'v2': yolo_postproc = yolo.yolov2_postproc
    elif args['yolo_version'] == 'v3': yolo_postproc = yolo.yolov3_postproc
    biases = bias_selector(args)

    import caffe
    caffe.set_mode_cpu()
    print(args)
    if(args['gpu'] is not None):
      caffe.set_mode_gpu()
      caffe.set_device(args['gpu'])

    net = caffe.Net(deploy_model, weights, caffe.TEST)

    net_h, net_w = net.blobs['data'].data.shape[-2:]
    args['net_h'] = net_h
    args['net_w'] = net_w

    for i,img in enumerate(images):
        if((i+1)%100 == 0): print(i+1, "images processed")
        raw_img, img_shape = xdnn_io.loadYoloImageBlobFromFile(img, net_h, net_w)

        net.blobs['data'].data[...] = raw_img
        out = net.forward()

        caffeOutput = sorted(out.values(), key=lambda item: item.shape[-1])
        boxes = yolo_postproc(caffeOutput, args, [img_shape], biases=biases)

        print("{}. Detected {} boxes in {}".format(i, len(boxes[0]), img))

        # Save the result
        boxes = boxes[0]
        if(args['results_dir']):
            filename = os.path.splitext(os.path.basename(img))[0]
            out_file_txt = os.path.join(args['results_dir'], filename + '.txt')
            print("Saving {} boxes to {}".format(len(boxes), out_file_txt)); sys.stdout.flush()
            saveDetectionDarknetStyle(out_file_txt, boxes, img_shape)
            if(args['visualize']):
                out_file_png = os.path.join(args['results_dir'], filename + '.png')
                print("Saving result to {}".format(out_file_png)); sys.stdout.flush()
                draw_boxes(img, boxes, labels, colors, out_file_png)
        # draw_boxes(images[i],bboxes,class_names,colors=[(0,0,0)]*num_classes)

    return len(images)


def main():
    parser = argparse.ArgumentParser()
    parser = yolo_parser_args(parser)
    parser.add_argument('--deploymodel', help="network definition prototxt file in case of caffe",
                        required=True, type=extant_file, metavar="FILE")
    parser.add_argument('--caffemodel', help="network weights caffe model file in case of caffe",
                        required=True, type=extant_file, metavar="FILE")
    parser.add_argument('--images', nargs='*',
        help='directory or raw image files to use as input', required=True, type=extant_file, metavar="FILE")
    parser.add_argument('--labels', help='label ID', type=extant_file, metavar="FILE")
    parser.add_argument('--golden', help='Ground truth directory', type=extant_file, metavar="FILE")
    parser.add_argument('--mean_value', type=int, nargs=3, default=[0,0,0],  # BGR for Caffe
        help='image mean values ')
    parser.add_argument('--pxscale', type=float, default=(1.0/255.0), help='pix cale value')
    parser.add_argument('--transpose', type=int, default=[2,0,1], nargs=3, help="Passed to caffe.io.Transformer function set_transpose, default 2,0,1" )
    parser.add_argument('--channel_swap', type=int, default=[2,1,0], nargs=3, help="Passed to caffe.io.Transformer function set_channel_swap, default 2,1,0")
    parser.add_argument('--caffe_backend_path', help='caffe backend')
    parser.add_argument('--gpu', type=int, default=None, help='GPU-ID to run Caffe inference on GPU')
    args = parser.parse_args()
    args = xdnn_io.make_dict_args(args)

    num_images_processed = yolo_gpu_inference(args['caffe_backend_path'],
                       args['images'],
                       args['deploymodel'],
                       args['caffemodel'],
                       args['results_dir'],
                       args['iouthresh'],
                       args['scorethresh'],
                       args['mean_value'],
                       args['pxscale'],
                       args['transpose'],
                       args['channel_swap'],
                       args['yolo_model'],
                       args['classes'], args)

    print('num images processed : ', num_images_processed)

    # mAP calculation
    if(args['golden']):
        labels = xdnn_io.get_labels(args['labels'])
        print()
        print("Computing mAP score  : ")
        print("Class names are  : {} ".format(labels))
        mAP = calc_detector_mAP(args['results_dir'], args['golden'], len(labels), labels, args['prob_threshold'], args['iouthresh'])
        sys.stdout.flush()

if __name__ == '__main__':
    main()
