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

import os, subprocess
import shutil

models = [
  "https://www.xilinx.com/bin/public/openDownload?filename=squeezenet.zip",
  "https://www.xilinx.com/bin/public/openDownload?filename=models.container.tensorflow.inception_v1_baseline.pb_2019-07-18.zip",
  "https://www.xilinx.com/bin/public/openDownload?filename=inception_v4.zip",
  "https://www.xilinx.com/bin/public/openDownload?filename=models.container.tensorflow.resnet50_baseline.pb_2019-07-18.zip",
  "https://www.xilinx.com/bin/public/openDownload?filename=resnet_v1_101.zip",
  "https://www.xilinx.com/bin/public/openDownload?filename=resnet_v1_152.zip",
]

# Where will we work
workDir = os.path.dirname(os.path.realpath(__file__)) + "/TEMP"

# Where are we putting models
modelsDir = os.path.dirname(os.path.realpath(__file__)) + "/models"

try:
  os.makedirs(modelsDir)
except OSError as e:
  if e.errno != os.errno.EEXIST:
    print("Error creating model directory, check permissions!")
    raise
  print ("Model directory already exists!")

try:
  os.makedirs(workDir)
except OSError as e:
  if e.errno != os.errno.EEXIST:
    print("Error creating work directory, check permissions!")
    raise
  print ("Work directory already exists!")

os.chdir(workDir)

for model in models:
  subprocess.call(["wget",model,"-O","temp.zip"])
  subprocess.call(["unzip","-o","temp.zip"])
  # Strip Unnecessary heirarchy
  for Dir,SubDirs,Files in os.walk("models"):
    if len(Files) > 0:
      break
  for File in Files:
    subprocess.call(["mv",os.path.join(Dir,File),modelsDir])
  subprocess.call(["rm","-rf","temp.zip","models"])

shutil.rmtree(workDir)
