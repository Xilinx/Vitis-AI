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


import warnings
import scipy
warnings.filterwarnings('ignore')
import matplotlib
import time#test time
import scipy.signal as signal#fftconvolve
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import caffe
import numpy as np
import os
from PIL import Image #for open image
from skimage.transform import resize
import math # for IPM

size = 1 
ratio = 8
def Find_local_maximum(datase):
    t = time.time()
    window = signal.general_gaussian(51, p=0.5, sig=1)
    peak_p = np.zeros([60, 80])
    peak_p = np.argmax(datase, 2)
    seed_x = [] #seed points for IPM
    seed_y = [] #seed points for IPM
    print("fft_convolve time 1 :", time.time() - t)
    #For better local maximum
    peak_p = np.max(datase, 2)
    print('shape:',np.shape(peak_p))
    t = time.time()
    for i in range(0, 60):
        peak_row = datase[i, :, :]
        if(sum(np.argmax(peak_row, 1)) == 0):
            continue
        j = 0
        while(j < 79):
            l = np.array([])
            while(np.argmax(peak_row[j]) > 0 and np.argmax(peak_row[j]) == np.argmax(peak_row[j+1]) and j < 79):
                l = np.append(l, max(peak_row[j]))
                j += 1
            j += 1
            if(len(l) > 0):
                l = np.append(l, max(peak_row[j]))
                #middle point
                max_idx = j - len(l.tolist()) // 2 - 1
                #local maximum
                #max_idx = j - np.where(l == max(l))[0]
                seed_y.append(max_idx)
                seed_x.append(i)
    print("Time of fftconvolve 2: ", time.time() - t)
    return np.array(seed_y), np.array(seed_x)

mean = np.array([105, 117, 123])
caffe.set_mode_gpu()
caffe.set_device(2)
model_def = "./float/test.prototxt"
model_weights = "./float/trainval.caffemodel"
#load model
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)

transformer = caffe.io.Transformer({'data': (net.blobs['data'].data).shape})
print((net.blobs['data'].data[0]).shape)

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mean)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

#caltech-lane
filename = open("./code/test/demo_caltech.txt")
test_file = [i.split() for i in filename.readlines()]
img_label = []
TP_sum_nn = np.array([0,0,0])
TP_FP_sum_nn = np.array([0,0,0])
TP_FN_sum_nn = np.array([0,0,0])
TP_sum_total = np.array([0,0,0])
TP_FP_sum_total = np.array([0,0,0])
TP_FN_sum_total = np.array([0,0,0])
fig_num = 0
for file_idx in range(0, len(test_file)):
    #path = "/home/chengming/chengming/VPGNet" + test_file[file_idx][0]
    path = "./"+test_file[file_idx][0]
    print(path)
    image = caffe.io.load_image(path) 

    transformed_image = transformer.preprocess('data', image)   

# copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image

### perform classification
    t0 = time.time()
    output = net.forward()
    print("Time of NN : ", time.time() - t0)
    datase = np.transpose(output['multi-label'][0],(1,2,0))

#-----visualiza network result--------
    img = Image.open(path)
    img = img.resize((640//ratio, 480//ratio),Image.ANTIALIAS)
# ---------- compute accuracy, recall, precision, F1-score--------
    img_label = []
    t0 = time.time()
    img_label = test_file[file_idx][2:]
    #print ("test_file:",test_file[file_idx][:2])
    TP_FN = np.array([0,0,0])
    k = 0
    while(k + 5 < len(img_label)):
        TP_FN[int(img_label[k+4]) - 1] += 1
        k += 5
    Stage_index=np.zeros(np.shape(img_label))
    TP = np.array([0,0,0])
    TP_FP = np.array([0,0,0])
    for i in range(0, 60):#640x512 use 64):
        for j in range (0,80) :
            if(np.argmax(datase[i,j,:]) > 0):
                index_y = (j-1)*size + 1; index_x = (i-1)*size + 1;
                class_ = np.argmax(datase[i, j, :])
                TP_FP[class_-1] += 1
                k = 0
                while(k + 5 < len(img_label)):
                    if(index_y >= (float(img_label[k])-1)/ratio and index_y <= float(img_label[k+2])/ratio and index_x >= (float(img_label[k+1])-1)/ratio and index_x  <= float(img_label[k+3])/ratio and class_ == int(img_label[k+4]) and Stage_index[k+4] == 0):
                        Stage_index[k+4] = 1
                        TP[class_-1] += 1
                        break
                    k += 5
    print("TP : ", TP)
    print("TP + FP : ", TP_FP)
    print("TP + FN : ", TP_FN)
    print("recall : ", TP.astype(float)/TP_FN.astype(float))
    TP_sum_nn += TP
    TP_FP_sum_nn += TP_FP
    TP_FN_sum_nn += TP_FN
    print("Time of computing recall of neural network: ", time.time() - t0)

#-------------------------------------------------- IPM ------------------------------------------
    t_start = time.time()
    seed_y, seed_x = Find_local_maximum(datase)
    
    t0 = time.time()
    TP = np.array([0,0,0])
    Stage_loc_index=np.zeros(np.shape(img_label))
    TP_FN_sum_total += TP_FN
    TP_FP = np.array([0,0,0])
    img_label = np.array(img_label)
    for i,v in enumerate(seed_x):
        TP_FP[np.argmax(datase[seed_x[i], seed_y[i]])-1] += 1
    
    for i,v in enumerate(seed_x):
        k = 0
        while(k + 5 < len(img_label)):
            if(seed_y[i] >= (float(img_label[k]) - 1)/(ratio) and seed_y[i] <= float(img_label[k+2])/(ratio) and seed_x[i] >= (float(img_label[k+1]) - 1)/(ratio) and seed_x[i] <= float(img_label[k+3])/(ratio) and np.argmax(datase[seed_x[i], seed_y[i]]) == int(img_label[k+4]) and Stage_loc_index[k+4]==0):
                Stage_loc_index[k+4] = 1
                TP[int(img_label[k+4])-1] += 1
                break
            k += 5

    TP_sum_total += TP
    TP_FP_sum_total += TP_FP
#    print("precision of after local maximum: ", TP.astype(float)/(TP_FP.astype(float)+1))
#    print("Time of computing recall of sample point", time.time() - t0)
    
Recall = TP_sum_nn.astype(float) / TP_FN_sum_nn.astype(float)
Precision = TP_sum_total.astype(float) / TP_FP_sum_total.astype(float)
print("Recall of local max : ", Recall)
print("Precision of local max : ", Precision)
print("F1-score if loacl max: ", 2 * Recall * Precision / (Recall + Precision))
print("Mean F1-score if loacl max: ", np.mean(2 * Recall * Precision / (Recall + Precision)))
