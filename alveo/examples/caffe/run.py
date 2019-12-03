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

from vai.dpuv1.tools.compile.bin.xfdnn_compiler_caffe  import CaffeFrontend as xfdnnCompiler
from decent import CaffeFrontend as xfdnnQuantizer
from vai.dpuv1.rt.scripts.framework.caffe.xfdnn_subgraph import CaffeCutter as xfdnnCutter

import numpy as np
import caffe

VAI_ALVEO_ROOT = os.getenv("VAI_ALVEO_ROOT", "../../")

# Generate scaling parameters for fixed point conversion
def Quantize(prototxt, caffemodel, test_iter=1, calib_iter=1, output_dir="work"):
  quantizer = xfdnnQuantizer(
    model = prototxt,
    weights = caffemodel,
    test_iter = test_iter,
    calib_iter = calib_iter,
    auto_test = True,
    output_dir = output_dir,
  )
  quantizer.quantize()

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
    networkfile = output_dir+"/deploy.prototxt",
    weights = output_dir+"/deploy.caffemodel",
    quant_cfgfile = output_dir+"/quantize_info.txt",
    generatefile = output_dir+"/compiler",
    quantz = output_dir+"/quantizer",
    **Getopts()
  )
  compiler.compile()

# Generate a new prototxt with custom python layer in place of FPGA subgraph
def Cut(prototxt,output_dir="work"):

  cutter = xfdnnCutter(
    cutAfter = "data", # Prototxt expected to have layer named "data"
    trainproto = prototxt, # train_val prototxt used to extract accuracy layers
    inproto = output_dir+"/deploy.prototxt",
    outproto = output_dir+"/xfdnn_auto_cut_deploy.prototxt",
    outtrainproto = output_dir+"/xfdnn_auto_cut_train_val.prototxt",
    xclbin = "/opt/xilinx/overlaybins/xdnnv3",
    netcfg = output_dir+"/compiler.json",
    quantizecfg = output_dir+"/quantizer.json",
    weights = output_dir+"/deploy.caffemodel_data.h5"
  )
  cutter.cut()

# Need to create derived class to clean up properly
class Net(caffe.Net):
  def __del__(self):
    for layer in self.layer_dict:
      if hasattr(self.layer_dict[layer],"fpgaRT"):
        del self.layer_dict[layer].fpgaRT

# Use this routine to evaluate accuracy on the validation set
def Infer(prototxt, caffemodel, numBatches=1):
  net = Net(prototxt, caffemodel, caffe.TEST)
  ptxtShape = net.blobs["data"].data.shape
  print ("Running with shape of: ",ptxtShape)
  results_dict = {}
  accum = {}
  for i in range(1, numBatches+1):
    out = net.forward()
    for k in out:
      if out[k].size != 1:
        continue
      if k not in accum:
        accum[k] = 0.0
      accum[k] += out[k]
      curAcc = out[k]
      avgAcc = accum[k] / i
      result = (k, " == This Batch: ", out[k], " Average: ", accum[k]/i, " Batch #: ", i)
      print("{} == This Batch: {:.4f} Average: {:.6f} Batch #: {}".format(k, float(out[k]), float(accum[k]/i), i))
      #print (*result)
      if k not in results_dict:
        results_dict[k] = []
      results_dict[k].append(result)
  print ("")
  return results_dict

# Use this routine to classify a single image
def Classify(prototxt,caffemodel,image,labels):
  net  = Net(prototxt, caffemodel, 1)
  data = net.blobs['data'].data
  image_dims = []
  image_dims.append(data.shape[2])
  image_dims.append(data.shape[3])
  classifier = caffe.Classifier(prototxt, caffemodel,
          image_dims, mean=np.array([104,117,123]),
          raw_scale=255, channel_swap=[2,1,0])
  predictions = classifier.predict([caffe.io.load_image(image)]).flatten()
  labels = np.loadtxt(labels, str, delimiter='\t')
  top_k = predictions.argsort()[-1:-6:-1]
  for l, p in zip(labels[top_k], predictions[top_k]):
    print ("{:.6f} {:s}".format(p, l))
  print ("")

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
  parser.add_argument('--validate_gpu', action="store_true", help='If validation is enabled, the model will be ran on the FPGA, and the validation set examined')
  parser.add_argument('--image', default=None, help='User can provide an image to run')
  args = vars(parser.parse_args())

  if args["prepare"]:
    if os.path.isdir(args["output_dir"]):
      print("Cleaning model artifacts in %s"%os.path.abspath(args["output_dir"]))
      filesToClean = [os.path.join(os.path.abspath(args["output_dir"]),f) for f in os.listdir(args["output_dir"])]
      for f in filesToClean:
        os.remove(f)
    Quantize(args["prototxt"], args["caffemodel"], args["qtest_iter"], args["qcalib_iter"], args["output_dir"])
    Compile(args["output_dir"])
    Cut(args["prototxt"], args["output_dir"])
    print("Generated model artifacts in %s" % os.path.abspath(args["output_dir"]))
    for f in os.listdir(args["output_dir"]):
      print("  "+f)

  # Both validate, and image require that the user has previously called prepare.
  if args["validate"]:
    Infer(args["output_dir"]+"/xfdnn_auto_cut_train_val.prototxt",args["output_dir"]+"/deploy.caffemodel",args["numBatches"])

  if args["image"]:
    Classify(args["output_dir"]+"/xfdnn_auto_cut_deploy.prototxt",args["output_dir"]+"/deploy.caffemodel",args["image"],"../deployment_modes/synset_words.txt")

  if args["validate_gpu"]:
    Infer(args["prototxt"],args["caffemodel"],args["numBatches"])
