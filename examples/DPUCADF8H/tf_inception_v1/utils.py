#!/usr/bin/env python
# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import os
import os.path as osp

from vai.dpuv1.rt.xdnn_io import loadImageBlobFromFileScriptBase

IMAGEROOT = osp.join(os.environ['HOME'], 'CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min')
IMAGELIST = osp.join(os.environ['HOME'], 'CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val.txt')
INPUT_NODES = 'input'

### Preprocessing formulas
### Available transformations: ['resize', 'resize2mindim', 'resize2maxdim', 'crop_letterbox',
###                             'crop_center', 'plot', 'pxlscale', 'meansub', 'chtranspose', 'chswap']

CMD_SEQ        = {
    'resnet50_v1_tf':[
        ('meansub', [103.94, 116.78, 123.68]),
        ('chswap',(2,1,0)),
        ('resize', [256, 256]),
        ('crop_center', [224, 224]),
        ],
    'inception_v1_tf':[
        ('pxlscale', 1/255.),
        ('meansub', 0.5),
        ('pxlscale', 2),
        ('resize', [256, 256]),
        ('crop_center', [224, 224]),
        ],
    'inception_v3_tf':[
        ('pxlscale', 1/255.),
        ('meansub', 0.5),
        ('pxlscale', 2),
        ('resize', [342, 342]),
        ('crop_center', [299, 299]),
        ],
    }

def load_image(iter, image_list = None, pre_process_function = None, input_nodes = 'data', get_shapes = False, batch_size = 1):
    images = []
    shapes = []
    labels = []
    for i in range(batch_size):
        image_name, label  = image_list[iter * batch_size + i].split(' ')
        print((image_name))
        image, shape = loadImageBlobFromFileScriptBase(image_name, pre_process_function)
        images.append(image)
        shapes.append(shape)
    if get_shapes:
        return {input_nodes: images}, shapes
    else:
        return {input_nodes: images}

def get_input_fn(pre_processing_function_name, imageroot = IMAGEROOT, imagelist = IMAGELIST, input_nodes = INPUT_NODES):
    cmd_seq = CMD_SEQ[pre_processing_function_name]
    with open(imagelist) as fin:
        lines = fin.readlines()
        image_list = list(map(lambda x:os.path.join(imageroot, x.strip()), lines))
    return partial(load_image, image_list = image_list, pre_process_function = cmd_seq, input_nodes = input_nodes)

for name in CMD_SEQ:
    globals()['input_fn_{}'.format(name)] = get_input_fn(name)
