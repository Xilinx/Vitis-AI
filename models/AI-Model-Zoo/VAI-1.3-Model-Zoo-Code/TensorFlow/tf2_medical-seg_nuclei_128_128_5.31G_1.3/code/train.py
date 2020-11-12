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

import tensorflow
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model

from lib.build_model import *
from lib.build_data import *
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
    parser.add_argument("--ckpt_path", type=str, default='float/',
                        help="Path to the save the trained weight file.")
    parser.add_argument("--resume_file", type=str, default=None,
                        help="resume the h5 file.")    
    # others                    
    parser.add_argument("--gpus", type=str, default='0',
                        help="choose gpu devices.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size for per-iteration training")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="base learning rate")    
    parser.add_argument("--epochs", type=int, default=100,
                        help="training epochs")
    # quantization config
    parser.add_argument("--quantize", type=bool, default=False,
                        help="whether do quantize or not.")       
    parser.add_argument("--quantize_output_dir", type=str, default='./quantized/',
                        help="directory for quantize output files.")                                     
    parser.add_argument("--dump", type=bool, default=False,
                        help="whether do dump or not.")       
    parser.add_argument("--dump_output_dir", type=str, default='./quantized/',
                        help="directory for dump output files.")                                     

    return parser.parse_args()

def main():
    args = get_arguments()

    for key, val in args._get_kwargs():
        print(key+' : '+str(val))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # Set data parameters
    NUM_CLASS=args.num_classes
    IMG_WIDTH, IMG_HEIGHT = map(int, args.input_size.split(','))
    IMG_CHANNELS = 3
    TRAIN_PATH = os.path.join(args.img_path, 'stage1_train/')
    TRAIN_VAL_SPLIT = 0.1
    # set model parameters
    seed = 42
    base_lr = args.learning_rate
    batch_size= args.batch_size
    epoches = args.epochs
    lr_deacy_setp = int(args.epochs * 0.3)
    ckpt_save_path = args.ckpt_path
    ckpt_name = 'model_unet'
    
    if not os.path.exists(ckpt_save_path):
        os.makedirs(ckpt_save_path)

    dir_path = ''

    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
    random.seed = seed
    np.random.seed = seed
    
    # get train_val data
    X_train, Y_train = get_train_val_data(dir_path, TRAIN_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)    
    # Build U-Net model
    if args.resume_file is not None:
        if args.dump:
          from tensorflow_model_optimization.quantization.keras import vitis_quantize
          with vitis_quantize.quantize_scope():
            model = load_model(args.resume_file)
        else:
          model = load_model(args.resume_file)
    else:
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        outputs = model_forward(inputs)
        model = Model(inputs=[inputs], outputs=[outputs])

    # Quantize
    if args.quantize:
        from tensorflow_model_optimization.quantization.keras import vitis_quantize
        quantizer = vitis_quantize.VitisQuantizer(model)
        model = quantizer.quantize_model(calib_dataset=X_train[0:1000])
        model.save(os.path.join(args.quantize_output_dir, 'quantized.h5'))
        print('Quantize finished, results in: {}'.format(args.quantize_output_dir))
        return

    # Dump
    if args.dump:
        quantizer = vitis_quantize.VitisQuantizer.dump_model(model, X_train[0:1], args.dump_output_dir)
        print('Dump finished, results in: {}'.format(args.dump_output_dir))
        return

    print(model.summary())

    # Build optimizer
    #learning_rate = tensorflow.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=base_lr, decay_steps=lr_deacy_setp, decay_rate=.1)

    #opt = tensorflow.keras.optimizers.SGD(learning_rate=base_lr)
    
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tensorflow.keras.metrics.MeanIoU(num_classes=NUM_CLASS)])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Fit model
    earlystopper = EarlyStopping(patience=15, verbose=1)
    checkpointer = ModelCheckpoint(os.path.join(ckpt_save_path, ckpt_name+'_model_weight.h5'), verbose=1, save_best_only=True)
    results = model.fit(X_train, Y_train, validation_split=TRAIN_VAL_SPLIT, batch_size=batch_size, epochs=epoches, 
                    callbacks=[earlystopper, checkpointer])

if __name__ == '__main__':
    main()
