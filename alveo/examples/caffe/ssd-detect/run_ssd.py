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
#coding=utf_8
import numpy as np
import cv2
import os
import sys

#from vai.dpuv1.tools.compile.bin.xfdnn_compiler_caffe  import CaffeFrontend as xfdnnCompiler
#from decent import CaffeFrontend as xfdnnQuantizer
import subprocess
from vai.dpuv1.rt.scripts.framework.caffe.xfdnn_subgraph import CaffeCutter as xfdnnCutter

import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import time
import argparse


def Quantize(prototxt,caffemodel,test_iter=1,calib_iter=1):
    os.environ["DECENT_DEBUG"] = "1"
    subprocess.call(["vai_q_caffe", "quantize",
                 "--model", prototxt,
                 "--weights", caffemodel,
                 "--calib_iter", str(calib_iter)])

# Standard compiler arguments for XDNNv3
def Getopts():
  return {
     "bytesperpixels":1,
     "dsp":96,
     "memory":9,
     "ddr":"256",
     "cpulayermustgo":True,
     "usedeephi":True,
  }

name = "inception_v2_ssd"
# Generate hardware instructions for runtime -> compiler.json
def Compile(prototxt="quantize_results/deploy.prototxt",\
            caffemodel="quantize_results/deploy.caffemodel",\
            quantize_info="quantize_results/quantize_info.txt"):
    subprocess.call(["vai_c_caffe",
                    "--prototxt", prototxt,
                    "--caffemodel", caffemodel,
                    "--net_name", name,
                    "--output_dir", "work",
                    "--arch", "/opt/vitis_ai/compiler/arch/dpuv1/ALVEO/ALVEO.json",
                    "--options", "{\"quant_cfgfile\":\"%s\", \
                    \"pipelineconvmaxpool\":False, \
                    }" %(quantize_info)])

# Generate a new prototxt with custom python layer in place of FPGA subgraph
def Cut(prototxt):
  cutter = xfdnnCutter(
    inproto="quantize_results/deploy.prototxt",
    trainproto=prototxt,
    outproto="xfdnn_auto_cut_deploy.prototxt",
    outtrainproto="xfdnn_auto_cut_train_val.prototxt",
    cutAfter="data",
    xclbin="/opt/xilinx/overlaybins/xdnnv3",
    netcfg="work/compiler.json",
    quantizecfg="work/quantizer.json",
    weights="work/weights.h5",
    #profile=True
  )
  cutter.cut()



##################### Mean and threshold configure #####################

view_theshold = 0.3
conf_theshold = 0.01

#################################################################################


font = cv2.FONT_HERSHEY_SIMPLEX
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]


def get_labelname(labelmap, labels):
    '''
    get labelname from lablemap and lables
    :param labelmap: map of label to name
    :param labels: label list
    :return: labelname list 
    '''
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in range(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

def declare_network(model_def, model_weights, labelmap_file, args):
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array(args["img_mean"]))
    transformer.set_input_scale('data', args["img_input_scale"])

    return net,transformer,labelmap

def detect_one_image(net,transformer,labelmap,image_path,image_resize_height, image_resize_width, mean, is_view=False, image_name=None, fp=None):
    '''
    detect one image use model
    :param image: image matrix
    :param image_name: image name
    :param fp: handle of results record file
    :return: None
    '''
    assert os.path.exists(image_path)
    image = cv2.imread(image_path)
    assert image is not None
    height, width = image.shape[0:2]
    image_resize = cv2.resize(image, (image_resize_width, image_resize_height))
    net.blobs['data'].reshape(1, 3, image_resize_height, image_resize_width)
    transformed_image = transformer.preprocess('data', image_resize)
    net.blobs['data'].data[...] = transformed_image
    start = time.time()
    detections = net.forward()['detection_out']
    end = time.time()
    print ("Foward time: ", end - start )
    det_label = detections[0, 0, :, 1]
    det_conf = detections[0, 0, :, 2]
    det_xmin = detections[0, 0, :, 3]
    det_ymin = detections[0, 0, :, 4]
    det_xmax = detections[0, 0, :, 5]
    det_ymax = detections[0, 0, :, 6]
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_theshold]
    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]
    if top_conf.size > 0:    
        size = len(top_labels)
        for i in range(size):
            f_xmin = width * top_xmin[i]
            int_xmin = int(f_xmin)
            f_ymin = height * top_ymin[i]
            int_ymin = int(f_ymin)
            f_xmax = width * top_xmax[i]
            int_xmax = int(f_xmax)
            f_ymax = height * top_ymax[i]
            int_ymax = int(f_ymax)
            if is_view and top_conf[i] >= view_theshold:
                color = colors_tableau[int(top_label_indices[i])]
                print  (str(top_labels[i]),':','xmin',int_xmin,'ymin',int_ymin,'xmax',int_xmax,'ymax', int_ymax) 
                cv2.rectangle(image, (int_xmin, int_ymin), (int_xmax, int_ymax), color, 1)
                cv2.putText(image, str(top_labels[i]), (int_xmin, int_ymin + 10), font, 0.4, (255, 255, 255), 1) 
                cv2.putText(image, str(top_conf[i]), (int_xmin, int_ymin - 10), font, 0.4, (255, 255, 255), 1)
            if image_name is not None and fp is not None:
                fp.writelines(image_name + " " + str(top_labels[i]) + " " + str(top_conf[i]) + " " \
                              + str(f_xmin) + " " + str(f_ymin) + " " + str(f_xmax) + " " + str(f_ymax) + "\n")  
    if is_view:
        cv2.imwrite("res_det.jpg", image)     


def compute_map_of_datset(net, transformer, lablemap, image_list_file, det_res_file, gt_file, test_image_root, image_resize_height, image_resize_width, mean, compute_map_script_path):
    '''
    compute map of dataset
    :param image_list_file:
    :param det_res_file:
    :param gt_file:
    :return: None
    '''  
    assert os.path.exists(image_list_file)
    f_image_list = open(image_list_file, 'r')
    lines = f_image_list.readlines()
    f_image_list.close()
    f_res_record = open(det_res_file, 'w')
    for line in lines:
        image_name = line.strip()
        image_path = test_image_root + image_name + '.jpg'
        detect_one_image(net, transformer, lablemap, image_path,image_resize_height, image_resize_width, mean, image_name=image_name, fp=f_res_record)
    f_res_record.close()
    os.system("python " +  compute_map_script_path + " -mode detection " +  \
              " -gt_file " + gt_file  +  " -result_file " + det_res_file \
              + " -detection_use_07_metric True")


def Detect(deploy_file, caffemodel, image,labelmap_file, args):
    net, transformer, labelmap = declare_network(deploy_file, caffemodel, labelmap_file, args)
    N, C, H, W = net.blobs['data'].data.shape
    detect_one_image(net, transformer, labelmap, image, H, W, np.array(args["img_mean"]), is_view=True)


def Infer(prototxt, caffemodel, args):
    net, transformer, labelmap = declare_network(prototxt, caffemodel, args["labelmap_file"], args)
    N, C, H, W = net.blobs['data'].data.shape
    compute_map_of_datset(net, transformer, labelmap, args["image_list_file"], args["det_res_file"], args["gt_file"], args["test_image_root"], H, W, np.array(args["img_mean"]), args["compute_map_script_path"])




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--prototxt', required=True, type=str,
                        help='Provide the xfdnn_auto_cut prototxt generated by subgraph or original deploy.prototxt')
    parser.add_argument('--caffemodel', required=True, type=str,
                        help = 'Provide the caffemodel file')
    parser.add_argument('--prepare', action="store_true", help='In prepare mode, model preperation will be perfomred = Quantize + Compile')

    parser.add_argument('--qtest_iter', type=int, default=1, help='User can provide the number of iterations to test the quantization')
    parser.add_argument('--qcalib_iter', type=int, default=1, help='User can provide the number of iterations to run the quantization')
    parser.add_argument('--labelmap_file', type=str, help='Provide the lablemap file')
    parser.add_argument('--image', type=str, help='Provide image path')
    parser.add_argument('--validate', action="store_true", help='If validation is enabled, the model will be ran on the FPGA, and the validation set examined')

    parser.add_argument('--image_list_file', type=str, help='Provide image_list_file')
    parser.add_argument('--test_image_root', type=str, help='images root directory')
    parser.add_argument('--det_res_file', type=str, default= 'ssd_det_res.txt', help='Provide detected result file')
    parser.add_argument('--gt_file', type=str, help='Ground truth file')
    parser.add_argument('--compute_map_script_path', default= './evaluation_py2.py', type=str, help='compute map script path')
    parser.add_argument('--img_mean',type=int, nargs=3, default=[104,117,123],  # BGR for Caffe
			help='image mean values ')
    parser.add_argument('--img_input_scale', type=float, default=1.0, help='image input scale value ')
   

    args = vars(parser.parse_args())
    
    if 	args["prepare"]:
        Quantize(args["prototxt"],args["caffemodel"], args["qtest_iter"], args["qcalib_iter"])
        Compile()
        Cut(args["prototxt"])

    if	args["image"]:
        Detect("xfdnn_auto_cut_deploy.prototxt", args["caffemodel"], args["image"], args["labelmap_file"], args)

    if	args["validate"]:
        if ((args["image_list_file"] == None) or (args["gt_file"] == None) or (args["test_image_root"] == None)):
            print ('Provide the arguments for image_list_file, gt_file and test_image_root')
            exit(0)
        Infer("xfdnn_auto_cut_deploy.prototxt",args["caffemodel"], args)



       
