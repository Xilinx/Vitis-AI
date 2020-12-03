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



# U-Net inference model on CPU

#%% import package

import numpy as np
import os
from skimage import color


import caffe
import cv2

import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

#%% define functions



def getImageArr( path , width , height ):  
    # load image
    img = plt.imread(path) # read image as 2d,scale: 0~1
    # resize, and convert 2d to 4d
    imarr = cv2.resize(img, (width , height),interpolation = cv2.INTER_AREA)
    imarr = color.gray2rgb(imarr) # 2d to 3d with 3 channels
    imarr = np.expand_dims(imarr, axis=0)    
    return imarr


def getSegmentationArr( path , nClasses,  width , height  ):  
    img = plt.imread(path) # read image as 2d, scale: 0~1
    img = cv2.resize(img, ( width , height),interpolation = cv2.INTER_NEAREST)

    seg_labels = np.zeros((  height , width  , nClasses ))
    for c in range(nClasses):
        seg_labels[: , : , c ] = (img == c ).astype(int)
    return np.expand_dims(seg_labels, axis=0)  





#%% dataset

dir_data = "./PhC-C2DH-U373/"        
dir_img = dir_data + "Img/"
dir_seg = dir_data + "Seg/"

file='test_list.txt' 
test_list = []   
with open(file,"r") as f:
    for line in f:
        test_list.append(line.strip())


#%% main with caffe model

# model configuration



model_def = './float/unet_U373_256.prototxt'
model_weights = './float/unet_U373_256.caffemodel'

#model_def = './quantize_results/deploy.prototxt'
#model_weights = './quantize_results/deploy.caffemodel'

#model_def = './quantize_results/quantize_train_test.prototxt'
#model_weights = './quantize_results/quantize_train_test.caffemodel'


net = caffe.Net(model_def, model_weights, caffe.TEST) 


#%% test model with 3 channel using train dataset with fixed threshold

# mean Intersection over Union
# Mean IoU = TP/(FN + TP + FP)
image_height , image_width = 256 , 256            
n_classes = 9


TP = np.zeros((n_classes,1))
FP = np.zeros((n_classes,1))
FN = np.zeros((n_classes,1))


for i in range(len(test_list)):  
    # load image    
    img = getImageArr(dir_img+test_list[i]+'.png' , image_width , image_height )
    # run net and take argmax for prediction
    net.blobs['input_1'].data[...] = np.transpose(img,(0,3,1,2)) # (B,C,H,W)
    net.forward()
    preds = net.blobs['outputs'].data
    predt = np.transpose(preds, (0,2,3,1))
    out = np.argmax(predt[0,:,:,:], axis=2) # segmentation  
    out = np.array(out, dtype=np.uint8)
    # load ground truth label
    seg = getSegmentationArr(dir_seg+test_list[i]+'.tif' ,n_classes, image_width , image_height  )
    label = np.argmax(seg[0,:,:,:], axis=2) # segmentation  
    label = np.array(label, dtype=np.uint8)  

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
print("_________________")
print("test Mean IoU: {:4.3f}".format(mIoU))



#%% execute a sample image to get masked output

if not os.path.exists('test_output'):
    os.mkdir('test_output')
    
image_height , image_width = 256 , 256            
n_classes = 9

# choose any file from test_list
i=0

# load image
img = getImageArr(dir_img+test_list[i]+'.png' , image_width , image_height )# read image as 2d,scale: 0~1
# run net and take argmax for prediction
net.blobs['input_1'].data[...] = np.transpose(img,(0,3,1,2)) # (B,C,H,W)
net.forward()
preds = net.blobs['outputs'].data
predt = np.transpose(preds, (0,2,3,1))
out = np.argmax(predt[0,:,:,:], axis=2) # segmentation  
out = np.array(out, dtype=np.uint8)


plt.figure(figsize = (15, 7))
plt.subplot(1,3,1)
plt.imshow(img[0,:,:,0],cmap='gray')
plt.title('original_resize')

plt.subplot(1,3,2)
plt.imshow(out)
plt.title('predicted mask')

plt.subplot(1,3,3)
plt.imshow(img[0,:,:,0],cmap='gray')
masked_imclass = np.ma.masked_where(out == 0, out)
plt.imshow(masked_imclass, alpha=0.5 )
plt.title('masked image')
# save file
plt.savefig('./test_output/'+test_list[i]+'_cpu.png')


    
    