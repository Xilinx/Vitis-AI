# semantic segmenation
# caffe model from deephi


#%% import packages
import numpy as np
from PIL import Image
import os
#import sys
#import glob
import cv2
#import matplotlib.pyplot as plt
#import time

import caffe

# Need to create derived class to clean up properly
class Net(caffe.Net):
  def __del__(self):
    for layer in self.layer_dict:
      if hasattr(self.layer_dict[layer],"fpgaRT"):
        del self.layer_dict[layer].fpgaRT

#%% define functions

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

def segment_output(net, img_file):
    IMG_MEAN = np.array((104,117,123))  # mean_values for B, G,R
    INPUT_W, INPUT_H = 512, 256 # W, H  512, 256
    TARGET_W, TARGET_H = 2048, 1024
    
    im_ = cv2.imread(img_file)
    w, h = TARGET_W, TARGET_H
    in_ = cv2.resize(im_, (INPUT_W, INPUT_H))
    in_ = in_ * 1.0
    in_ -= IMG_MEAN
    in_ = in_.transpose((2,0,1))

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)

    #save color_output
    pred_label_color = label_img_to_color(out)
    color_to_save = Image.fromarray(pred_label_color.astype(np.uint8))
    color_to_save = color_to_save.resize((w,h))
    return color_to_save

def segment(net, img_file):
    IMG_MEAN = np.array((104,117,123))  # mean_values for B, G,R
    INPUT_W, INPUT_H = 512, 256 # W, H  512, 256
#    TARGET_W, TARGET_H = 2048, 1024
    
    im_ = cv2.imread(img_file)
#    w, h = TARGET_W, TARGET_H
    in_ = cv2.resize(im_, (INPUT_W, INPUT_H))
    in_ = in_ * 1.0
    in_ -= IMG_MEAN
    in_ = in_.transpose((2,0,1))

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)

    return out

#%% main 
    
# model configuration for cpu
model_def = './quantize_results/deploy.prototxt'
model_weights = './quantize_results/deploy.caffemodel'

net_cpu = Net(model_def, model_weights, caffe.TEST) 


# model configuration for fpga
model_def = 'xfdnn_deploy.prototxt'
model_weights = './quantize_results/deploy.caffemodel'
net_fpga = Net(model_def, model_weights, caffe.TEST) 


# file path
#img_path = './leftImg8bit/val/frankfurt/'
img_path = './cityscapes/val/photo/'
output_path = './fpga_output/'

if not os.path.exists('fpga_output'):
    os.mkdir('fpga_output')
    
image_list = os.listdir(img_path)

print('segmentation images will be stored in fpga_output folder.')
## create segmenation image
for img in image_list[:30]:   
    print(img)    
    # create segmentation image
    semantic_seg = segment_output(net_fpga, img_path+img)
    # save output
    semantic_seg.save(output_path+img)
        
   
    
## calculate mIoU
# mean Intersection over Union
# Mean IoU = TP/(FN + TP + FP)
n_classes = 19

TP = np.zeros((n_classes,1))
FP = np.zeros((n_classes,1))
FN = np.zeros((n_classes,1))

print('calculate mIoU between cpu output and fpga output')
for img in image_list[:100] :  
    print(img)
    # segmentation output from fpga model
    out = segment(net_fpga, img_path+img)

    # segmentation output from cpu model
    label = segment(net_cpu, img_path+img)

    for c in range(n_classes):
        TP[c] += np.sum( (label == c)&(out==c) )
        FP[c] += np.sum( (label != c)&(out==c) )
        FN[c] += np.sum( (label == c)&(out != c)) 
        
IoUs = []
for c in range(n_classes):
    IoU = TP[c]/float(TP[c] + FP[c] + FN[c])
    print("class {:02.0f}: #TP={:6.0f}, #FP={:6.0f}, #FN={:5.0f}, IoU={:4.3f}".format(c,int(TP[c]),int(FP[c]),int(FN[c]),float(IoU)))
    IoUs.append(float(IoU))
mIoU = np.mean(IoUs)
del net_fpga
print("_________________")
print("mean IoU: {:4.3f}".format(mIoU))
