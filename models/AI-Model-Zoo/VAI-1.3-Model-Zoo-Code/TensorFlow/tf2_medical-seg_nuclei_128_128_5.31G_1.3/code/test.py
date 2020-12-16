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

import tensorflow 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model

from lib.build_data import get_train_val_data
from PIL import Image
import argparse

def get_arguments():
    """Parse all the arguments.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="TF2 Semantic Segmentation")

    parser.add_argument("--input_size", type=str, default='128,128', help="Input shape: [H, W]")
    #data config
    parser.add_argument("--img_path", type=str, default='./data/nuclei_data',
                        help="Path to the directory containing the cityscapes validation images.")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes to predict.")
    # model config
    parser.add_argument("--weight_file", type=str, default='float/weights.h5',
                        help="Path to the final best weights.")
    # others                    
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--save_path", type=str, default='./results_visulization/',
                        help="where to save the vis results.")
    parser.add_argument("--return_seg", type=bool, default=True,
                        help="resturn gray prediction")
    # quantization config
    parser.add_argument("--quantize", type=bool, default=False,
                        help="whether do quantize or not.")       

    return parser.parse_args()

def main():
    args = get_arguments()

    for key, val in args._get_kwargs():
        print(key+' : '+str(val))

    # Set data parameters
    NUM_CLASS=args.num_classes
    IMG_WIDTH, IMG_HEIGHT = map(int, args.input_size.split(','))
    IMG_CHANNELS = 3
    TRAIN_PATH = os.path.join(args.img_path, 'stage1_train/')
    TRAIN_VAL_SPLIT = 0.1
    # set model parameters
    seed = 42
    ckpt_file = args.weight_file

    # save prediction
    save_pred = args.return_seg
    output_path = args.save_path

    dir_path = ''

    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
    random.seed = seed
    np.random.seed = seed
    
    # load model
    if args.quantize:
      from tensorflow_model_optimization.quantization.keras import vitis_quantize
      with vitis_quantize.quantize_scope():
        model = load_model(ckpt_file)
    else:
      model = load_model(ckpt_file)

    # load data
    X_train, Y_train = get_train_val_data(dir_path, TRAIN_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    index = int(X_train.shape[0]*float(1-TRAIN_VAL_SPLIT))
    X_val = X_train[index:]
    Y_val = Y_train[index:]
    
    model.compile(metrics=['accuracy', tensorflow.keras.metrics.MeanIoU(num_classes=2)])
    results = model.evaluate(X_val, Y_val, batch_size=1)

    if save_pred:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        preds_val = model.predict(X_val, verbose=1)
        preds_val_t = (preds_val > 0.5).astype(np.uint8)

        for ix in range(len(preds_val_t)): 
            #image = Image.fromarray(X_train[int(X_train.shape[0]*0.9):][ix])
            #image.save(output_path + '/img_'+str(ix)+'.png')

            #gt = np.squeeze(Y_train[int(X_train.shape[0]*0.9):][ix])
            #gt = Image.fromarray(gt)
            #gt.save(output_path+'/gt_'+str(ix)+'.png')

            pred_val = np.squeeze(preds_val_t[ix])
            pred_val[pred_val==1] = 255
            pred_val = Image.fromarray(pred_val)
            pred_val.save(output_path+'/pred_'+str(ix)+'.png')


if __name__ == '__main__':
    main()
