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

import requests
import json

def request(rest_api_url="http://localhost:9001/predict", image_path="/home/mluser/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/ILSVRC2012_val_00000001.JPEG"):

  image = open(image_path, "rb").read()
  payload = {"image": image}

  # submit the request
  r = requests.post(rest_api_url, files=payload).json()


  # ensure the request was sucessful
  if r["success"]:
    print(json.dumps(r, indent=4, sort_keys=True))

  # otherwise, the request failed
  else:
    print("Request failed")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='pyXFDNN')
  parser.add_argument('--rest_api_url', default="", help='Url to the REST API eg: http://localhost:9000/predict')
  parser.add_argument('--image_path', default="", help='Path to the image eg: /home/mluser/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/ILSVRC2012_val_00000001.JPEG')

  args = vars(parser.parse_args())

  if args["rest_api_url"] and args["image_path"]:
    request(args["rest_api_url"], args["image_path"])
  else:
    print("Missing arguments, provide --rest_api url <>  and --image_path <>")
