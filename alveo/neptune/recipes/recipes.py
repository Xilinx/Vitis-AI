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

import neptune.recipes.recipe as Recipe

root_path = os.environ['VAI_ALVEO_ROOT']
model_path = root_path + '/artifacts'
artifact_server = "http://www.xilinx.com/bin/public/openDownload?filename="
if os.path.isdir('/opt/xilinx/overlaybins/xdnnv3'):
    xclbin_path =  '/opt/xilinx/overlaybins/xdnnv3'
else:
    xclbin_path = root_path + '/overlaybins/xdnnv3'

if not os.path.exists(model_path):
    os.makedirs(model_path)

def add_facedetect(graph, previous_node):
    args = {
        'artifact' : {'url': artifact_server + 'facedetect.zip','path':model_path,'key':'facedetect'},
        'batch_sz': 1,
        'outtrainproto': None,
        'input_names': ['data'],
        'cutAfter': 'data',
        'outproto': 'xfdnn_deploy.prototxt',
        'xdnnv3': True,
        'inproto': model_path + '/deploy.prototxt',
        'profile': False,
        'trainproto': None,
        'weights': model_path + '/facedetect/weights.h5',
        'netcfg': model_path + '/facedetect/compiler.json',
        'quantizecfg': model_path + '/facedetect/quantizer.json',
        'xclbin': xclbin_path,
        'output_names': ['pixel-conv', 'bb-output'],
        'PE': 0,
        'numprepproc': 4,
        'numstream': 8,
        'img_input_scale': 1.0,
        'img_raw_scale': 255.0
    }

    graph.add_node("facedetect", args)
    graph.add_edge(previous_node, "facedetect")
    return graph

def add_coco(graph, previous_node):
    args_pre = {
        'netcfg': model_path + '/yolov2/compiler.json',
        'preprocseq': [
            ('resize2maxdim',(224,224)),
            ('pxlscale',(1.0/255.0)),
            ('crop_letterbox',(0.5)),
            ('chtranspose',(2,0,1)),
            ('chswap',(2,1,0))
        ],
        'img_mean':  [0.0, 0.0, 0.0],
        'img_input_scale': 1.0,
        'img_raw_scale': 255.0,
        'numprepproc': 4,
        'numstream': 8,
        'benchmarkmode': False
    }
    args_fpga = {
        'netcfg': model_path + '/yolov2/compiler.json',
        'weights': model_path + '/yolov2/weights.h5',
        'artifact' : {'url': artifact_server + 'yolov2.zip','path':model_path,'key':'yolov2'},
        'xclbin': xclbin_path,
        'quantizecfg': model_path + '/yolov2/quantizer.json',
        'numprepproc': 4,
        'numstream': 8,
        'batch_sz': 1,
        'PE': 0
    }
    args_post = {
        'netcfg': model_path + '/yolov2/compiler.json',
        'labels': model_path + '/yolov2/labels.txt',
        'zmqpub': False,
        'golden': None,
        'batch_sz': 1,
        'classes': 80,
        'anchor_boxes': 5,
        'scorethresh': 0.24,
        'iouthresh': 0.3,
        'benchmarkmode': False,
        'servermode': True
    }

    graph.add_node("pre", args_pre)
    graph.add_node("fpga", args_fpga)
    graph.add_node("yolopost", args_post)
    graph.add_edge(previous_node, "pre")
    graph.add_edge("pre", "fpga")
    graph.add_edge("fpga", "yolopost")
    graph.add_edge("yolopost", "fpga")
    return graph

def add_imagenet(graph, previous_node):
    args_pre = {
        'netcfg': model_path + '/resnet50/compiler.json',
        'preprocseq': [
            ('resize', (224, 224)),
            ('meansub', [104.007, 116.669, 122.679]),
            ('chtranspose', (2, 0, 1))
        ],
        'img_mean':  [104.007, 116.669, 122.679],
        'img_input_scale': 1.0,
        'img_raw_scale': 255.0,
        'numprepproc': 4,
        'numstream': 8,
        'benchmarkmode': False
    }
    args_fpga = {
        'netcfg': model_path + '/resnet50/compiler.json',
        'weights': model_path + '/resnet50/weights.h5',
        'artifact' : {'url': artifact_server + 'resnet50.zip','path':model_path,'key':'resnet50'},
        'xclbin': xclbin_path,
        'quantizecfg': model_path + '/resnet50/quantizer.json',
        'numprepproc': 4,
        'numstream': 8,
        'batch_sz': 1,
        'PE': 0
    }
    args_post = {
        'netcfg': model_path + '/resnet50/compiler.json',
        'weights': model_path + '/resnet50/weights.h5',
        'batch_sz': 1,
        'outsz': 1000,
        'labels': model_path + '/resnet50/labels.txt',
        'golden': None,
        'zmqpub': False,
        'benchmarkmode': False,
        'servermode': True
    }

    graph.add_node("fpga", args_fpga)
    graph.add_node("pre", args_pre)
    graph.add_node("post", args_post)
    graph.add_edge(previous_node, "pre")
    graph.add_edge("pre", "fpga")
    graph.add_edge("fpga", "post")
    graph.add_edge("post", "fpga")
    return graph

def recipe_ping():
    graph = Recipe.Graph()
    graph.add_node("ping")
    handlers = Recipe.Handlers('handlers.ping_handler')
    return Recipe.Construct(graph, 'ping', 'ping', handlers)

def recipe_facedetect():
    graph = Recipe.Graph()
    graph.add_node("image_frontend")
    graph = add_facedetect(graph, "image_frontend")
    handlers = Recipe.Handlers('handlers.image_handler', 'handlers.image_handler')
    return Recipe.Construct(graph, 'face', 'face', handlers)

def recipe_sface():
    graph = Recipe.Graph()
    graph.add_node("stream", {'frequency': 10, 'timeout': 15000})
    graph = add_facedetect(graph, "stream")
    graph.set_arg('facedetect', 'callback', True)
    handlers = Recipe.Handlers('handlers.video_handler', 'handlers.video_handler')
    return Recipe.Construct(graph, 'sface', 'sface', handlers)

def recipe_imagenet():
    graph = Recipe.Graph()
    graph.add_node("image_frontend")
    graph = add_imagenet(graph, "image_frontend")
    handlers = Recipe.Handlers('handlers.image_handler', 'handlers.image_handler')
    return Recipe.Construct(graph, 'imagenet', 'imagenet', handlers)

def recipe_simnet():
    graph = Recipe.Graph()
    graph.add_node("stream", {'frequency': 10, 'timeout': 15000})
    graph = add_imagenet(graph, "stream")
    graph.set_arg('post', 'callback', True)
    handlers = Recipe.Handlers('handlers.video_handler', 'handlers.video_handler')
    return Recipe.Construct(graph, 'simnet', 'simnet', handlers)

def recipe_coco():
    graph = Recipe.Graph()
    graph.add_node("image_frontend")
    graph = add_coco(graph, "image_frontend")
    handlers = Recipe.Handlers('handlers.image_handler', 'handlers.image_handler')
    return Recipe.Construct(graph, 'coco', 'coco', handlers)

def recipe_scoco():
    graph = Recipe.Graph()
    graph.add_node("stream", {'frequency': 10, 'timeout': 15000})
    graph = add_coco(graph, "stream")
    graph.set_arg('yolopost', 'callback', True)
    handlers = Recipe.Handlers('handlers.video_handler', 'handlers.video_handler')
    return Recipe.Construct(graph, 'scoco', 'scoco', handlers)
