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
import random
import warnings

import numpy as np
#import pandas as pd

from PIL import Image
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from PIL import ImageFile

# Get train and test IDs
def get_train_val_data(dir_path, TRAIN_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    train_ids = next(os.walk(TRAIN_PATH))[1]
    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    print('Getting and resizing train images and masks ... ')
    
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        #Read image files iteratively
        path = TRAIN_PATH + id_
        img = imread(os.path.join(dir_path, path + '/images/' + id_ + '.png'))[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        img /= 255.0
        #Append image to numpy array for train dataset
        X_train[n] = np.array(img)
    
        #Read corresponding mask files iteratively
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    
        #Looping through masks
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            #Read individual masks
            mask_ = imread(os.path.join(dir_path, path + '/masks/' + mask_file))
            #Expand individual mask dimensions
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1)
        
            #Overlay individual masks to create a final mask for corresponding image
            mask = np.maximum(mask, mask_)
    
        #Append mask to numpy array for train dataset
        Y_train[n] = mask
    print('Done!')
    return X_train, Y_train

def get_test_data(dir_path, TEST_PATH, FINAL_TEST_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    test_ids = next(os.walk(TEST_PATH))[1]
    final_test_ids = next(os.walk(FINAL_TEST_PATH))[1]
    # Get and resize test images
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    sizes_test = []

    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        #Read images iteratively
        img = imread(dir_path + path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        #Get test size
        sizes_test.append([img.shape[0], img.shape[1]])
        #Resize image to match training data
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        #Append image to numpy array for test dataset
        X_test[n] = img

    print('Done!')
    return X_test