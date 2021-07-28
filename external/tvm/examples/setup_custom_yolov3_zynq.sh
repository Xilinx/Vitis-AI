#!/bin/bash

# Copyright 2021 Xilinx Inc.
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

# SETUP POST-PROCESSING FOR CUSTOM YOLOV3
pip3 install easydict

if [ -d /home/root/tensoflow-yolov3 ]; then
  rm -rf  /home/root/tensoflow-yolov3
fi

git clone https://github.com/YunYang1994/tensorflow-yolov3 /home/root/tensorflow-yolov3
sed -i 's/import tensorflow as tf/ /g' /home/root/tensorflow-yolov3/core/utils.py
sed -i 's/.\//\/home\/root\/tensorflow-yolov3\//' /home/root/tensorflow-yolov3/core/config.py
