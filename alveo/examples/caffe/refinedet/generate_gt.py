#coding=utf_8
import os
import sys
from argparse import ArgumentParser
import numpy as np
import cv2

def parse_args():
    parser = ArgumentParser(description="ssd evaluation on coco2014-person")
    parser.add_argument('--caffe-root', '-c', type=str, 
            default='../../../../../caffe/',
            help='path to caffe root')
    parser.add_argument('--data-root', '-d', type=str,
            default='./coco2014/Images/',
            help='path to validation images')
    parser.add_argument('--image-list', '-i', type=str,
            default='./coco2014/val2014.txt',
            help='path to image list file')
    parser.add_argument('--anno-root', '-a', type=str,
            default='./coco2014/Annotations',
            help='path to annotations')
    parser.add_argument('--gt-file', '-gt', type=str,
            default='./gt_file.txt',
            help='file record test image annotations.')
    parser.add_argument('--det-file', '-det', type=str,
            default='./det_file.txt',
            help='file record test image detection results.')
    parser.add_argument('--prototxt', '-p', type=str,
            default='../../float/test.prototxt',
            help='path to caffemodel prototxt')
    parser.add_argument('--weights','-w', type=str,
            default='../../float/trainval.caffemodel',
            help='path to caffemodel weights')
    parser.add_argument('--labelmap', '-l', type=str,
            default='../../labelmap.prototxt',
            help='path to labelmap file')
    parser.add_argument('--eval-script-path', '-e', type=str,
            default='./evaluation_py2.py',
            help='path to eval map script')
    return parser.parse_args()

args = parse_args()

def generate_gt_file(anno_root, image_list, gt_file):
    with open(gt_file, 'w') as fw_gt:
         with open(image_list, 'r') as fr_image:
             imagename_list = fr_image.readlines()
             for imagename in imagename_list:
                 anno_path = os.path.join(anno_root, imagename.strip() + '.txt')
                 fr_anno = open(anno_path, 'r')
                 anno_lines = fr_anno.readlines()
                 fr_anno.close()
                 for anno in anno_lines:
                    fw_gt.writelines(anno)   


if __name__ == "__main__":
    print("Generating ground-truth file `{}` for {} with annotations in {}"
              .format(args.gt_file, args.image_list, args.anno_root))
    generate_gt_file(args.anno_root, args.image_list, args.gt_file)

