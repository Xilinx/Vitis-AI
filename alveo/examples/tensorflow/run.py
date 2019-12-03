#!/usr/bin/env python
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

from __future__ import print_function

import os, sys, argparse
import subprocess
import tensorflow as tf
import numpy as np
import cv2

# Bring in VAI Quantizer, Compiler, and Partitioner
from vai.dpuv1.rt.xdnn_rt_tf import TFxdnnRT as xdnnRT
from vai.dpuv1.rt.xdnn_util import dict2attr, list_depth, make_list
from utils import get_input_fn, top5_accuracy, LABEL_OFFSET, BATCH_SIZE
from shutil import rmtree


# Environment Variables (obtained by running "source overlaybins/setup.sh")
if 'VAI_ALVEO_ROOT' in os.environ and os.path.isdir(os.path.join(os.environ['VAI_ALVEO_ROOT'], 'overlaybins/xdnnv3')):
    XCLBIN = os.path.join(os.environ['VAI_ALVEO_ROOT'], 'overlaybins/xdnnv3')
else:
    XCLBIN = '/opt/xilinx/overlaybins/xdnnv3/'

def get_default_compiler_args():
    return {
        'dsp':                  96,
        'memory':               9,
        'bytesperpixels':       1,
        'ddr':                  256,
        'data_format':          'NHWC',
        'mixmemorystrategy':    True,
        'pipelineconvmaxpool':  True,
        'xdnnv3':               True,
        'usedeephi':            True,
        'pingpongsplit':        True,
        'deconvinfpga':         True,
        'quantz':               '',
        'fcinfpga':             True,
        'bridges': ['bytype', 'Concat'],
    }



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='pyXFDNN')
  parser.add_argument('--model', default="", help='User must provide the network model file')
  parser.add_argument('--input_nodes', default="", help='User must provide the network input names [comma seperated with no spacing]')
  parser.add_argument('--output_nodes', default="", help='User must provide the network output names [comma seperated with no spacing]')
  parser.add_argument('--input_shapes', default="", help='User must provide the network input shapes [comma seperated with no spacing]')
  parser.add_argument('--output_dir', default="work", help='Optionally, save all generated outputs in specified folder')
  parser.add_argument('--pre_process', default="", help='pre-process function for quantization calibration')
  parser.add_argument('--label_offset', default=LABEL_OFFSET, help='Optionally, label offset of the dataset')
  parser.add_argument('--batch_size', default=BATCH_SIZE, help='batch sizes to run')
  parser.add_argument('--quantize', action="store_true", default=False, help='In quantize mode, model will be Quantize')
  parser.add_argument('--validate_cpu', action="store_true", help='If validation_cpu is enabled, the model will be validated on cpu')
  parser.add_argument('--validate', action="store_true", help='If validation is enabled, the model will be partitioned, compiled, and ran on the FPGA, and the validation set examined')
  parser.add_argument('--c_input_nodes', default=None, help='Compiler input node names')
  parser.add_argument('--c_output_nodes', default=None, help='Compiler output node names')
  args = dict2attr(parser.parse_args())

  if not args.pre_process:
    raise ValueError('please provide --pre_process input. Valid option: resnet50, inception_v1, inception_v3, inception_v4, squeezenet')

  if args.quantize:
    from tensorflow import GraphDef
    from tensorflow.contrib import decent_q

    input_graph_def = GraphDef()
    with open(args.model, "rb") as f:
      input_graph_def.ParseFromString(f.read())

    if os.path.isdir(args.output_dir):
      print('Cleaning model artifacts in {}'.format(os.path.abspath(args.output_dir)))
      filesToClean = [os.path.join(os.path.abspath(args.output_dir),f) for f in os.listdir(args.output_dir)]
      for f in filesToClean:
        if os.path.isfile(f):
          os.remove(f)
        elif os.path.isdir(f):
          rmtree(f)
    else:
      os.makedirs(args.output_dir)

    input_node_names = make_list(args.input_nodes)
    output_node_names = make_list(args.output_nodes)
    input_shapes = make_list(args.input_shapes)
    input_shapes = [input_shapes] if list_depth(input_shapes) == 1 else input_shapes

    if 'VAI_ALVEO_ROOT' in os.environ and os.path.isdir(os.path.join(os.environ['VAI_ALVEO_ROOT'], 'vai/dpuv1')):
      arch_json = os.path.join(os.environ['VAI_ALVEO_ROOT'], 'vai/dpuv1/tools/compile/bin/arch.json')
    elif 'VAI_ROOT' in os.environ:
      arch_json = os.path.join(os.environ['VAI_ROOT'], 'compiler/arch/dpuv1/ALVEO/ALVEO.json')
    else:
      arch_json = '/opt/vitis-ai/compiler/arch/dpuv1/ALVEO/ALVEO.json'
    input_fn = get_input_fn(args.pre_process, args.input_nodes)
    q_config = decent_q.QuantizeConfig(input_nodes = input_node_names,
        output_nodes = output_node_names,
        input_shapes = input_shapes,
        output_dir = args.output_dir,
        method= 1,
        calib_iter = 500 // args.batch_size)
    decent_q.quantize_frozen(input_graph_def, input_fn, q_config)

    #subprocess.call(['vai_q_tensorflow', 'inspect',
    #                 '--input_frozen_graph', args.model])
    #subprocess.call(['vai_q_tensorflow', 'quantize',
    #                 '--input_frozen_graph', args.model,
    #                 '--input_nodes', input_node_names,
    #                 '--output_nodes', output_node_names,
    #                 '--input_shapes', '{},{},{},{}'.format(*args.input_shapes),
    #                 '--output_dir', args.output_dir,
    #                 '--input_fn', 'utils.input_fn',
    #                 '--method', '1',
    #                 '--calib_iter', '{}'.format(500 // args.batch_size)])

    subprocess.call(['vai_c_tensorflow',
                     '--frozen_pb', args.output_dir+'/deploy_model.pb',
                     '--arch', arch_json,
                     '--output_dir', args.output_dir,
                     '--net_name', args.output_dir+'/fix_info.txt',
                     '--quant_info'])

    print("Generated model artifacts in %s"%os.path.abspath(args["output_dir"]))
    for f in os.listdir(args["output_dir"]):
      print("  "+f)

  if args.validate_cpu:
    iter_cnt = 500 // args.batch_size

    tf.reset_default_graph()
    with open(args.model, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      tf.import_graph_def(graph_def, name='')

    graph = tf.get_default_graph()
    top5_accuracy(graph, args.input_nodes, args.output_nodes, iter_cnt, args.batch_size, args.pre_process, args.label_offset)

  if args.validate:
    iter_cnt = 500 // args.batch_size

    ### Partition and compile
    rt = xdnnRT(None,
                networkfile=args.model,
                quant_cfgfile=args.output_dir+'/fix_info.txt',
                batch_sz=args.batch_size,
                startnode=args.c_input_nodes,
                finalnode=args.c_output_nodes,
                xclbin=XCLBIN,
                device='FPGA',
                placeholdershape="{\'%s\': [%d,%d,%d,%d]}" % (args.input_nodes, args.batch_size, args.input_shapes[1], args.input_shapes[2], args.input_shapes[3]),
                savePyfunc=True,
                **get_default_compiler_args()
               )

    ### Accelerated execution
    ## load the accelerated graph
    graph = rt.load_partitioned_graph()

    top5_accuracy(graph, args.input_nodes, args.output_nodes, iter_cnt, args.batch_size, args.pre_process, args.label_offset)
