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


import os,subprocess
import shutil

"""
def substitute(File,Replacee,Replacement):

  # Read in the file
  with open(File, 'r') as f :
    filedata = f.read()

  # Replace the target string
  filedata = filedata.replace(Replacee,Replacement)

  # Write the file out again
  with open(File, 'w') as f:
    f.write(filedata)

  return 0
"""

models = [
  "https://www.xilinx.com/bin/public/openDownload?filename=models.container.caffe.bvlc_googlenet_2019-05-02_12_32.zip",
   "https://www.xilinx.com/bin/public/openDownload?filename=models.container.caffe.inception_v2_2019-05-02_12_32.zip",
   "https://www.xilinx.com/bin/public/openDownload?filename=models.container.caffe.inception_v3_2019-05-02_12_32.zip",
   "https://www.xilinx.com/bin/public/openDownload?filename=models.container.caffe.inception_v4_2019-12-2_10-22.zip",
   "https://www.xilinx.com/bin/public/openDownload?filename=models.container.caffe.resnet50_v1_2019-05-02_12_32.zip",
   "https://www.xilinx.com/bin/public/openDownload?filename=models.container.caffe.resnet50_v2_2019-05-02_12_32.zip",
   "https://www.xilinx.com/bin/public/openDownload?filename=models.container.caffe.squeezenet_2019-05-02_12_32.zip",
   "https://www.xilinx.com/bin/public/openDownload?filename=models.container.caffe.vgg16_2019-05-02_12_32.zip",
   "https://www.xilinx.com/bin/public/openDownload?filename=models.container.caffe.inception_v2_ssd_2019-05-06_0765.zip",
   "https://www.xilinx.com/bin/public/openDownload?filename=FPN_CityScapes.zip",
   "https://www.xilinx.com/bin/public/openDownload?filename=maps_BtoA.zip",
   "https://www.xilinx.com/bin/public/openDownload?filename=maps_AtoB.zip",
   "https://www.xilinx.com/bin/public/openDownload?filename=cityscapes_BtoA.zip",
   "https://www.xilinx.com/bin/public/openDownload?filename=cityscapes_AtoB.zip",
   "https://www.xilinx.com/bin/public/openDownload?filename=facades_BtoA.zip",
   "https://www.xilinx.com/bin/public/openDownload?filename=bw2color.zip",
   
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
      pathToParent = Dir
      break
  subprocess.call(["mv",pathToParent,modelsDir])
  subprocess.call(["rm","-rf","temp.zip","models"])

shutil.rmtree(workDir)
