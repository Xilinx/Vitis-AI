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
from google import protobuf as pb
from google.protobuf import text_format
from caffe.io import caffe_pb2 as cpb2

def load_deploy_to_netparam(deploy_file):
  netparam = cpb2.NetParameter()
  with open(deploy_file, 'r') as f:
    text_format.Merge(f.read(), netparam)

  return netparam


def modify_input_dims(netparam, dims):
  ''' dims : [1, C, H, W]
  '''

  # Make sure first layer is a Input Layer"
  assert netparam.layer[0].type == "Input", "First layer is not Input Layer, Is this correct deploy.prototxt?"

  # Remove old shape & transform_param
  netparam.layer[0].input_param.shape[0].Clear()
  if netparam.layer[0].transform_param.yolo_height > 0: netparam.layer[0].transform_param.yolo_height=dims[-2]
  if netparam.layer[0].transform_param.yolo_width > 0: netparam.layer[0].transform_param.yolo_width=dims[-1]

  for i in dims:
    netparam.layer[0].input_param.shape[0].dim.append(i)

  return netparam


def store_netparam_to_deploy(deploy_file, netparam):
  with open(deploy_file, 'w') as f:
    f.write(text_format.MessageToString(netparam))


if __name__ == "__main__":
  import sys
  import argparse
  parser = argparse.ArgumentParser(description="Tool to modify input dimensions in the deploy prototxt.")
  parser.add_argument('--in_shape', nargs=3, type=int,
          help='input dimensions in CxHxW, for eg: --in_shape=3 224 224')
  parser.add_argument('--input_deploy_file', type=str,
          help='Input deploy file')
  parser.add_argument('--output_deploy_file', type=str,
          help='Output deploy file')

  args = parser.parse_args()

  shape = [1,] + args.in_shape

  netparam = load_deploy_to_netparam(args.input_deploy_file)
  new_net = modify_input_dims(netparam, shape)
  store_netparam_to_deploy(args.output_deploy_file, new_net)
