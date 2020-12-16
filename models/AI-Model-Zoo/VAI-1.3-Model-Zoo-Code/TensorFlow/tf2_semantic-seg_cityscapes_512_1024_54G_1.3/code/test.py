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
import numpy as np
import argparse
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import cv2
from tqdm import tqdm
import os
from glob import glob
from PIL import Image
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.applications.resnet50 import preprocess_input

#from code.models.erfnet import erfnet

def get_arguments():
    """Parse all the arguments.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="TF2 Semantic Segmentation")

    parser.add_argument('--arch', default='cbr', choices=['cbr', 'crb'], \
                         help='model arch design: cbr=conv+bn+relu, crb=conv+relu+bn')
    parser.add_argument("--input_size", type=str, default='512,1024', help="Input shape: [H, W]")
    #data config
    parser.add_argument("--img_path", type=str, default='./data/cityscapes/val_images/',
                        help="Path to the directory containing the cityscapes validation images.")
    parser.add_argument("--num_classes", type=int, default=20,
                        help="Number of classes to predict.")
    # model config
    parser.add_argument("--weight_file", type=str, default='float/weights.h5',
                        help="Path to the final best weights.")
    # others                    
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--save_path", type=str, default='./results_visulization_erfnet/512x1024/',
                        help="where to save the vis results.")
    parser.add_argument("--return_seg", type=bool, default=True,
                        help="resturn gray prediction")
    parser.add_argument("--add_color", type=bool, default=False,
                        help="merge corlors masks on the RGB images")
    # quantization config
    parser.add_argument("--quantize", type=bool, default=False,
                        help="whether do quantize or not.")       

    return parser.parse_args()

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
        19: [0, 0, 0],
        }

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]
            img_color[row, col] = np.array(label_to_color[label])
    return img_color



def segment(args, model, image, name=None):
    global b
    alpha = 0.5
    dims = image.shape
    H, W = image.shape[0], image.shape[1]
    if not os.path.exists(args.save_path):
       os.makedirs(args.save_path)
    
    raw_img = image.copy()
    h, w = map(int, args.input_size.split(','))
    image = cv2.resize(image, (w, h))
    x = image.copy()
    x = preprocess_input(np.expand_dims(x, axis=0))
    x = x / 255.0
    z = model.predict(x)
    z = np.squeeze(z)
    y = np.argmax(z, axis=2)
    out = cv2.resize(y, dsize=(W, H), interpolation=cv2.INTER_NEAREST)

    if args.return_seg:
        cv2.imwrite(os.path.join(args.save_path, name), out)
    if args.add_color:
        color_mask = label_img_to_color(out)
        color_mask = Image.fromarray(color_mask.astype(np.uint8)).convert('RGB')
        color_mask.save(os.path.join(args.save_path, 'color_'+ name))

def main():
    args = get_arguments()

    if args.arch=='cbr':
        from code.models.erfnet_cbr import erfnet
    else:
        from code.models.erfnet_crb import erfnet

    for key, val in args._get_kwargs():
        print(key+' : '+str(val))
  
    h, w = map(int, args.input_size.split(','))
    num_classes = args.num_classes

    if args.quantize:
      from tensorflow_model_optimization.quantization.keras import vitis_quantize
      with vitis_quantize.quantize_scope():
        model = load_model(args.weight_file)
    else:
      #  model = erfnet(input_shape=(h, w, 3), num_classes=num_classes)
      model = load_model(args.weight_file)

    print(model.summary())

    image_dir = args.img_path
    image_list = os.listdir(image_dir)
    image_list.sort()
    print('{} frames found'.format(len(image_list)))

    for i in tqdm(range(len(image_list))):
        image_file = image_list[i]
        image = load_img(os.path.join(image_dir, image_file))
        image = img_to_array(image)
        segment(args, model, image, image_list[i])

if __name__ == '__main__':
    main()
