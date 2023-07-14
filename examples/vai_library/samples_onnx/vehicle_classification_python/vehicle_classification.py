# Copyright 2022-2023 Advanced Micro Devices Inc.
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

import argparse
import numpy as np
import onnxruntime
import warnings
warnings.filterwarnings('ignore')
from PIL import Image, ImageDraw, ImageFont
import numpy as np

mean = [0, 0, 0]
scale = [0.00392156863, 0.00392156863, 0.00392156863]

def get_batch():
  import subprocess
  cmd =  "xdputil query | grep 'DPU Batch' | awk -F':' '{ print $2}' | awk -F','  '{ print $1}' "
  p = subprocess.Popen(cmd,  stdout=subprocess.PIPE, shell=True)
  ret = p.communicate()
  if ret[0] == b'':
      return 1
  return int(ret[0])

class OnnxSession:
    def __init__(self, onnx_model_path):
        self.session = onnxruntime.InferenceSession(onnx_model_path, providers=["VitisAIExecutionProvider"], 
                                                    provider_options=[{"config_file":"/usr/bin/vaip_config.json"}] )

    def input_shape(self):
        return self.session.get_inputs()[0].shape

    def run(self, input_data):
        input_name = self.session.get_inputs()[0].name
        raw_result = self.session.run([], {input_name: input_data})
        return raw_result

class InputData:
    def __init__(self, image_file_path, shape):
        self.shape = shape
        self.data = Image.open(image_file_path)

    def preprocess(self):
        shape = self.shape
        resize_data = self.data.resize((shape[2], shape[3]))
        tran_data = np.array(resize_data).transpose(2, 0, 1)
        float_data = tran_data.astype('float32')
        norm_img_data = np.zeros(float_data.shape).astype('float32')
        for i in range(float_data.shape[0]):
            norm_img_data[i,:,:] = (float_data[i,:,:] - mean[i]) * scale[i]
        norm_img_data = norm_img_data.astype('float32')
        return norm_img_data

def softmax(res):
    x = np.array(res)
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    res_list =  (e_x / e_x.sum(axis=0)).tolist()
    return res_list

def sort_idx(res):
    return np.flip(np.squeeze(np.argsort(res)))

def main():
    parser = argparse.ArgumentParser(description="argparser")
    parser.add_argument("--image_file_path", default="./sample_vehicleclassification.jpg")
    # note, it supports multiple pics, set it to "./sample_1.jpg ./sample_2.jpg"
    parser.add_argument("--onnx_model_path", default="./vehicle_type_resnet18_onnx_pt.onnx")
    parser.add_argument("--class_file_path", default="./vehicle_type.txt")

    image_file_path = parser.parse_args().image_file_path
    onnx_model_path = parser.parse_args().onnx_model_path
    class_file_path = parser.parse_args().class_file_path

    fsp = image_file_path.split()
    real_batch = min( get_batch(), len(fsp))
    onnx_session = OnnxSession(onnx_model_path)
    model_shape = onnx_session.input_shape()
    # print("model_shape :", model_shape ) #  ['ResNet::input_0_dynamic_axes_1', 3, 224, 224]
    image_data = []
    for i in range(0, real_batch):
        image_data.append( InputData(fsp[i], model_shape).preprocess() )
    # print("image_data ", len(image_data), len(image_data[0]),  len(image_data[0][0]), len(image_data[0][0][0])) # 2 3 224 224
    raw_result = onnx_session.run(image_data)
    # print("raw_result : " , len(raw_result),  len(raw_result[0]), len(raw_result[0][0]) ) # float array of [1, 2, 7] [array([[-13.75 , 5.258 ...  ]]float32]

    with open(class_file_path, "rt") as f:
        classes = f.read().rstrip('\n').split('\n')

    for i in range(0, real_batch):
        res_list = softmax(raw_result[0][i])
        sortedx = sort_idx(res_list)

        print('============ Top 5 labels for input ', i, ' are: =======================')
        for k in sortedx[:5]:
            print(classes[k], res_list[k])

if __name__ == "__main__":
    main()

