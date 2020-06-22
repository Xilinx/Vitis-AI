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



#coding:utf-8
import sys
from sys import argv
import os
import numpy as np
import itertools
import pdb
from numpy import *
from sklearn.decomposition import PCA
import scipy.io as sio
import multiprocessing
import cv2
import time
import argparse

parser = argparse.ArgumentParser(description='Face Recognition Test')
parser.add_argument('--caffe_path', type=str, default='/path/to/your/caffe_xilinx',
            help='your caffe path')
parser.add_argument('--model', type=str, default='',
            help='caffe model for test')
parser.add_argument('--prototxt', type=str, default='',
            help='caffe model prototxt for test')
parser.add_argument('--testset', type=str, default='3007',
            help='evaluation dataset')
parser.add_argument('--gpu', type=int, default=0,
            help='gpu card for test, should be 0')
parser.add_argument('--batch_size', type=int, default=64,
            help='batch size for test, larger for faster')
parser.add_argument('--feat_name', type=str, default='BatchNorm_61',
            help='output feature name')
parser.add_argument('--height', type=int, default=112,
            help='image height')
parser.add_argument('--width', type=int, default=96,
            help='image width')
parser.add_argument('--feat_dim', type=int, default=512,
            help='output feature dimension')

args = parser.parse_args()
sys.path.insert(0,os.path.join(args.caffe_path,'python'))
sys.path.insert(0,os.path.join(args.caffe_path,'python','caffe'))

import caffe

def getRecongnationRate(PdisSorted, NdisSorted, fpr):
    positiveNum = len(PdisSorted)
    PDIS = np.array(PdisSorted)
    tpr = []
    threshold = []
    for i in range(len(fpr)):
        thresh = NdisSorted[int(len(NdisSorted)*fpr[i])]
        threshold.append(thresh)
        TPNUM = PDIS[PDIS>=thresh].shape[0]
        tpr.append(TPNUM/float(positiveNum)) 
    return tpr, threshold

def cosDisFast(identificationID, identificationLife, featureID, featureLife, featDim):
    A = featureLife
    C = featureID
    
    #normalization  
    AMOD = linalg.norm(A,axis=1)
    CMOD = linalg.norm(C,axis=1)
    AMOD = AMOD.reshape(AMOD.shape[0], 1)
    CMOD = CMOD.reshape(CMOD.shape[0], 1)
    A = A*1.0/AMOD
    C = C*1.0/CMOD
    
    A = A.reshape(A.shape[0],featDim)
    C = C.reshape(C.shape[0],featDim)

    CT = np.transpose(C)
    F = dot(A,CT)
    pair = map(list,itertools.product(identificationLife,identificationID))
    pair_list = [value for value in pair]
    Life_ID = np.array(pair_list)
    #Life_ID = np.array(map(list,itertools.product(identificationLife,identificationID)))
    mask = Life_ID[:,0] == Life_ID[:,1]
    mask = mask.reshape(F.shape[0],F.shape[1])

    Pdis = F[mask].tolist()
    Ndis = F[~mask].tolist()
    
    #print('Positive pair number: %s'%len(Pdis))
    #print('Negative pair number: %s'%len(Ndis))
    PdisSorted = sorted(Pdis,reverse=True)
    NdisSorted = sorted(Ndis,reverse=True)
    return PdisSorted, NdisSorted



def featureExtraction(partSize, deployFile, caffemodelPath, testImgListID, testImgListLife, BATCH, featName, featDim):
    mean = np.zeros((3, partSize[0], partSize[1]))
    mean[:,:,:] = 127.5
    featureID = np.zeros((0,featDim))
    featureLife = np.zeros((0,featDim))
    fopenID = open(testImgListID,'r')
    fopenLife = open(testImgListLife,'r')
    imgPathsID = fopenID.readlines()
    imgPathsLife = fopenLife.readlines()
    lastNumID = len(imgPathsID)%BATCH
    lastNumLife = len(imgPathsLife)%BATCH
    caffe_model= caffemodelPath 
    net=caffe.Net(deployFile, caffe_model, caffe.TEST)
    transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
    transformer.set_transpose('data',(2,0,1))
    transformer.set_mean('data',mean)
    transformer.set_input_scale('data',0.0078125)
    net.blobs['data'].reshape(BATCH,3,partSize[0],partSize[1])
    identificationID = []
    for i in range(0,len(imgPathsID),BATCH):
        for j in range(BATCH):
            identificationID.append(imgPathsID[i+j].strip().split('/')[-1].split('_')[0])
            imgName = imgPathsID[i+j].strip()
            #image=caffe.io.load_image(imgName,1,False)
            
            flags = cv2.IMREAD_COLOR
            img = cv2.imread(imgName, flags)
            image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            image = np.array(image)
            image = np.float64(image.reshape((partSize[0],partSize[1],3)))
            net.blobs['data'].data[j] = transformer.preprocess('data',image)
            if (i+j) == len(imgPathsID)-1:
                break
        outIDtmp = net.forward(end=featName)
        outID = outIDtmp[featName].reshape((BATCH,featDim))
        if (i+j) != len(imgPathsID)-1 or lastNumID==0:
            featureID = concatenate((featureID,outID),axis=0)
        else:
            featureID = concatenate((featureID,outID[0:lastNumID,:]),axis=0)
    identificationLife = []
    for i in range(0,len(imgPathsLife),BATCH):
        for j in range(BATCH):
            identificationLife.append(imgPathsLife[i+j].strip().split('/')[-1].split('_')[0])
            imgName = imgPathsLife[i+j].strip()
            #image=caffe.io.load_image(imgName,1,False)
            
            flags = cv2.IMREAD_COLOR
            img = cv2.imread(imgName, flags)
            image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            
            image = np.array(image)
            image = np.float64(image.reshape((partSize[0],partSize[1],3)))
            
            net.blobs['data'].data[j] = transformer.preprocess('data',image)
            if (i+j) == len(imgPathsLife)-1:
                break
        outLifetmp = net.forward(end=featName)
        outLife = outLifetmp[featName].reshape((BATCH,featDim))
        if (i+j) != len(imgPathsLife)-1 or lastNumLife==0:
            featureLife = concatenate((featureLife,outLife),axis=0)
        else:
            featureLife = concatenate((featureLife,outLife[0:lastNumLife,:]),axis=0)
    fopenID.close()
    fopenLife.close()
    return identificationID, identificationLife, featureID, featureLife

if __name__ == "__main__":

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)
    TESTSET = args.testset
    BATCH = args.batch_size
    caffemodelPath = args.model
    deployFile = args.prototxt
    partSize = [args.height, args.width]
    featName = args.feat_name
    featDim = args.feat_dim

    if TESTSET == '3007':
        testImgListID='./data/3007/3007_ID.txt'
        testImgListLife='./data/3007/3007_Life.txt'
    
    fpr = [1e-7,1e-6,1e-5,1e-4]
        
    identificationID, identificationLife, featureID, featureLife = featureExtraction(partSize, deployFile, caffemodelPath, testImgListID, testImgListLife, BATCH, featName, featDim)
    featureIDAll = featureID
    featureLifeAll = featureLife
        
    PdisSorted, NdisSorted = cosDisFast(identificationID, identificationLife, featureIDAll, featureLifeAll, featDim)
    tpr, Thresh = getRecongnationRate(PdisSorted, NdisSorted, fpr)
    
    print('Model: %s'%caffemodelPath)
    print('Test set: %s'%TESTSET)
    print('FPR\t TPR\t Thr')
    for k in range(len(fpr)):
        print('%.0e\t %.1f\t %.3f'%(fpr[k], tpr[k]*100, Thresh[k]))
    
    
