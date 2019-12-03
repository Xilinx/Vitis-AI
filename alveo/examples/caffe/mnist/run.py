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

import os,sys,argparse
import subprocess

from vai.dpuv1.tools.compile.bin.xfdnn_compiler_caffe  import CaffeFrontend as xfdnnCompiler
from xfdnn_subgraph import CaffeCutter as xfdnnCutter

import numpy as np
import caffe

VAI_ALVEO_ROOT = os.getenv("VAI_ALVEO_ROOT","../../")
MLSUITE_PLATFORM = os.getenv("MLSUITE_PLATFORM","1525")

# Generate scaling parameters for fixed point conversion
def Quantize(prototxt,caffemodel,test_iter=1,calib_iter=1,output_dir="work"):
  subprocess.call([
    'vai_q_caffe',
    'quantize',
    '-model', prototxt,
    '-weights', caffemodel,
    '-test_iter', test_iter,
    '-calib_iter', calib_iter,
    '-auto_test', True,
    '-output_dir', output_dir])

# Standard compiler arguments for XDNNv3
def Getopts():
  return {
     "bytesperpixels":1,
     "dsp":96,
     "memory":9,
     "ddr":"256",
     "cpulayermustgo":True,
     "forceweightsfullyconnected":True,
     "mixmemorystrategy":True,
     "pipelineconvmaxpool":True,
     "usedeephi":True,
  }

# Generate hardware instructions for runtime -> compiler.json
def Compile(output_dir="work"):

  compiler = xfdnnCompiler(
    networkfile=output_dir+"/deploy.prototxt",
    weights=output_dir+"/deploy.caffemodel",
    quant_cfgfile=output_dir+"/quantize_info.txt",
    generatefile=output_dir+"/compiler",
    quantz=output_dir+"/quantizer",
    **Getopts()
  )
  compiler.compile()

# Generate a new prototxt with custom python layer in place of FPGA subgraph
def Cut(prototxt,output_dir="work"):

  cutter = xfdnnCutter(
    cutAfter="data", # Prototxt expected to have layer named "data"
    trainproto=prototxt, # train_val prototxt used to extract accuracy layers
    inproto=output_dir+"/deploy.prototxt",
    outproto=output_dir+"/xfdnn_auto_cut_deploy.prototxt",
    outtrainproto=output_dir+"/xfdnn_auto_cut_train_val.prototxt",
    xclbin=VAI_ALVEO_ROOT+"/overlaybins/"+MLSUITE_PLATFORM+"/overlay_4.xclbin",
    netcfg=output_dir+"/compiler.json",
    quantizecfg=output_dir+"/quantizer.json",
    weights=output_dir+"/deploy.caffemodel_data.h5",
    profile=True
  )
  cutter.cut()

# Use this routine to evaluate accuracy on the validation set
def Infer(prototxt,caffemodel,numBatches=1):
  net = caffe.Net(prototxt,caffemodel,caffe.TEST)
  ptxtShape = net.blobs["data"].data.shape
  print ("Running with shape of: ",ptxtShape)
  results_dict = {}
  accum = {}
  for i in xrange(1,numBatches+1):
    out = net.forward()
    for k in out:
      if out[k].size != 1:
        continue
      if k not in accum:
        accum[k] = 0.0
      accum[k] += out[k]
      result = (k, " -- This Batch: ",out[k]," Average: ",accum[k]/i," Batch#: ",i)
      print (*result)
      if k not in results_dict:
        results_dict[k] = []
      results_dict[k].append(result)
  return results_dict

# Use this routine to classify a single image
def Classify(prototxt,caffemodel,image,labels,color=False):
  classifier = caffe.Classifier(prototxt,caffemodel)
  predictions = classifier.predict([caffe.io.load_image(image,color=color)]).flatten()
  labels = np.loadtxt(labels, str, delimiter='\t')
  top_k = predictions.argsort()[-1:-6:-1]
  for l,p in zip(labels[top_k],predictions[top_k]):
    print (l," : ",p)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='pyXFDNN')
  parser.add_argument('--prototxt', default="", help='User must provide the train_val prototxt')
  parser.add_argument('--caffemodel', default="", help='User must provide the caffemodel')
  parser.add_argument('--output_dir', default="work", help='Optionally, save all generated outputs in specified folder')
  parser.add_argument('--numBatches', type=int, default=1, help='User must provide number of batches to run')
  parser.add_argument('--qtest_iter', type=int, default=1, help='User can provide the number of iterations to test the quantization')
  parser.add_argument('--qcalib_iter', type=int, default=1, help='User can provide the number of iterations to run the quantization')
  parser.add_argument('--prepare', action="store_true", help='In prepare mode, model preperation will be perfomred = Quantize + Compile')
  parser.add_argument('--validate', action="store_true", help='If validation is enabled, the model will be ran on the FPGA, and the validation set examined')
  parser.add_argument('--validate_cpu', action="store_true", help='If validation is enabled, the model will be ran on the CPU, and the validation set examined')
  parser.add_argument('--image', default=None, help='User can provide an image to run')
  parser.add_argument('--image_cpu', default=None, help='User can provide an image to run on cpu')
  args = vars(parser.parse_args())

  # When using this switch, we will prepare the model for deployment on FPGA
  # This is aggregating the three steps of quantize, compile, and subgraph cutting
  # quantization determines the required scalars for mapping the FP32 weights to INT8
  # compilation determines the schedule of hardware level computations, by traversing the data flow graph
  # subgraph cutting is generating a new caffe prototxt where the bulk of the computation is replaced by a custom FPGA layer
  #   This is a convienient way to run pre an post processing in the CPU with caffe, and accelerate the core of the network on the FPGA.
  #   Being able to drive the full mechanism with caffe is also convienient when comparing CPU runs to FPGA runs, for accuracy verification
  if args["prepare"]:
    if os.path.isdir(args["output_dir"]):
      print("Cleaning model artifacts in %s"%os.path.abspath(args["output_dir"]))
      filesToClean = [os.path.join(os.path.abspath(args["output_dir"]),f) for f in os.listdir(args["output_dir"])]
      for f in filesToClean:
        os.remove(f)
    Quantize(args["prototxt"],args["caffemodel"],args["qtest_iter"],args["qcalib_iter"],args["output_dir"])
    Compile(args["output_dir"])
    Cut(args["prototxt"],args["output_dir"])
    print("Generated model artifacts in %s"%os.path.abspath(args["output_dir"]))
    for f in os.listdir(args["output_dir"]):
      print("  "+f)

  # Both validate, and image require that the user has previously called prepare.

  # Run the model on the FPGA. It will run a batch of images, and report accuracy. The images are pointed to by the prototxt.
  if args["validate"]:
    Infer(args["output_dir"]+"/xfdnn_auto_cut_train_val.prototxt",args["output_dir"]+"/deploy.caffemodel",args["numBatches"])

  # Run the original model on the CPU. It will run a batch of images, and report accuracy
  if args["validate_cpu"]:
    Infer(args["prototxt"],args["caffemodel"],args["numBatches"])

  # Run a single image on the FPGA
  if args["image"]:
    Classify(args["output_dir"]+"/xfdnn_auto_cut_deploy.prototxt",args["output_dir"]+"/deploy.caffemodel",args["image"],"mnist_words.txt")

  # Run a single image on the CPU
  if args["image_cpu"]:
    Classify(args["output_dir"]+"/deploy.prototxt",args["output_dir"]+"/deploy.caffemodel",args["image_cpu"],"mnist_words.txt")
