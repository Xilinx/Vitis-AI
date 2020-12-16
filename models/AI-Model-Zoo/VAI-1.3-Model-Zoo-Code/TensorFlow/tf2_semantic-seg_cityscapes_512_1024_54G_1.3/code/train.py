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
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.models import Model, load_model
import argparse
#from models.erfnet_cbr import erfnet

print('TensorFlow', tf.__version__)
global H, W, R_H, R_W

H, W = 512, 1024
R_H, R_W = 512, 1024

def get_arguments():
    """Parse all the arguments.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="TF2 Semantic Segmentation")

    parser.add_argument("--model_name", type=str, default='ERFNet', help="model name")
    parser.add_argument('--arch', default='cbr', choices=['cbr', 'crb'], \
                         help='model arch design: cbr=conv+bn+relu, crb=conv+relu+bn')
    #data config
    parser.add_argument("--data_path", type=str, default='./data/cityscapes_20cls/',
                        help="Path to the cityscapes director.")
    parser.add_argument("--num_classes", type=int, default=20,
                        help="Number of classes to predict.")
    # model optimization parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Path to the final best weights.")
    parser.add_argument("--learning_rate", type=float, default=1e-2,
                        help="base learning rate.")       
    parser.add_argument("--epochs", type=int, default=200,
                        help="Epoch numbers.")                                     
    # others                    
    parser.add_argument("--gpus", type=str, default='0',
                        help="choose gpu devices.")
    parser.add_argument("--ckpt_path", type=str, default='./float/',
                        help="where to save the vis results.")
    parser.add_argument("--resume", type=str, default=None,
                        help="resume weights file.")
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


def get_image(image_path, mask=False, flip=0):
    img = tf.io.read_file(image_path)
    if not mask:
        img = tf.cast(tf.image.decode_png(img, channels=3), dtype=tf.float32)
        img = tf.image.resize(images=img, size=[R_H, R_W])
        img = tf.image.random_brightness(img, max_delta=50.)
        img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
        img = tf.image.random_hue(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
        img = tf.clip_by_value(img, 0, 255)
        img = tf.case([(tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))
                      ], default=lambda: img)
        img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68])  # reuse caffe resnet mean_value
        img = img[:, :, ::-1] / 255.0
    else:
        img = tf.image.decode_png(img, channels=1)
        img = tf.cast(tf.image.resize(images=img, size=[R_H, R_W]), dtype=tf.uint8)
        img = tf.case([(tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))
                      ], default=lambda: img)
    return img



def load_data(image_path, mask_path):
    flip = tf.random.uniform(
        shape=[1, ], minval=0, maxval=2, dtype=tf.int32)[0]
    image, mask = get_image(image_path, flip=flip), get_image(
        mask_path, mask=True, flip=flip)
    return image, mask



def main(): 
    args = get_arguments()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    if args.arch=='cbr':
        from code.models.erfnet_cbr import erfnet
    else:
        from code.models.erfnet_crb import erfnet
        
    for key, val in args._get_kwargs():
        print(key+' : '+str(val))
    num_classes = args.num_classes
    batch_size = args.batch_size

    if not os.path.exists(os.path.join(args.ckpt_path, args.model_name)):
        os.makedirs(os.path.join(args.ckpt_path, args.model_name))

    image_list = sorted(glob(os.path.join(args.data_path,'train_images/*')))
    mask_list = sorted(glob(os.path.join(args.data_path,'train_masks/*')))

    val_image_list = sorted(glob(os.path.join(args.data_path,'val_images/*')))
    val_mask_list = sorted(glob(os.path.join(args.data_path,'val_masks/*')))

    print('Found', len(image_list), 'training images')
    print('Found', len(val_image_list), 'validation images')
    for i in range(len(image_list)):
        assert image_list[i].split('/')[-1].split('_leftImg8bit')[0] == mask_list[i].split('/')[-1].split('_gtFine_trainids')[0]

    for i in range(len(val_image_list)):
        assert val_image_list[i].split('/')[-1].split('_leftImg8bit')[0] == val_mask_list[i].split('/')[-1].split('_gtFine_trainids')[0]

    train_dataset = tf.data.Dataset.from_tensor_slices((image_list,mask_list))
    train_dataset = train_dataset.shuffle(buffer_size=128)
    train_dataset = train_dataset.apply(tf.data.experimental.map_and_batch(map_func=load_data,
                                       batch_size=batch_size,
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                       drop_remainder=True))
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_list,val_mask_list))
    val_dataset = val_dataset.apply(tf.data.experimental.map_and_batch(map_func=load_data,
                                       batch_size=batch_size,
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                       drop_remainder=True))
    val_dataset = val_dataset.repeat()
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)


    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope():
    if args.resume is not None:
        if args.dump:
          from tensorflow_model_optimization.quantization.keras import vitis_quantize
          with vitis_quantize.quantize_scope():
            model = load_model(args.resume)
        else:
          model = load_model(args.resume)
    else:
        model = erfnet(input_shape=(H, W, 3), num_classes=num_classes, l2=None)

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)

    # Quantize
    if args.quantize:
        from tensorflow_model_optimization.quantization.keras import vitis_quantize
        quantizer = vitis_quantize.VitisQuantizer(model)
        model = quantizer.quantize_model(calib_dataset=val_dataset.take(20))
        model.save(os.path.join(args.quantize_output_dir, 'quantized.h5'))
        print('Quantized finished, results in: {}'.format(args.quantize_output_dir))
        return

    # Dump
    if args.dump:
        quantizer = vitis_quantize.VitisQuantizer.dump_model(model, val_dataset.take(1), args.dump_output_dir)
        print('Dump finished, results in: {}'.format(args.dump_output_dir))
        return

    model.compile(loss=loss, optimizer=tf.optimizers.Adam(learning_rate=args.learning_rate), 
                  metrics=['accuracy'])


    mc = ModelCheckpoint(mode='min', filepath=os.path.join(args.ckpt_path, 'weights.h5'),
                     monitor='val_loss',
                     save_best_only='True',
                     save_weights_only='True', verbose=1)

    bc = ModelCheckpoint(mode='min', filepath=os.path.join(args.ckpt_path, 'model_weights.h5'),
                     monitor='val_loss',
                     save_best_only='True',
                     save_weights_only='False', verbose=1)


    callbacks = [mc, bc]
    model.fit(train_dataset,
          steps_per_epoch=len(image_list) // batch_size,
          epochs=args.epochs,
          validation_data=val_dataset,
          validation_steps=len(val_image_list) // batch_size,
          callbacks=callbacks)

if __name__ == '__main__':
    main()
