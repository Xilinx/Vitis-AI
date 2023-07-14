#!/usr/bin/env python
#
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

import os, sys
import shutil
import argparse
import subprocess

# Device artifacts
u50lv_v3e = {
  "tf_resnet_v1_50" : "https://www.xilinx.com/bin/public/openDownload?filename=resnet_v1_50_tf-u50lv-DPUCAHX8H-r2.5.0.tar.gz",
  "densebox_320_320" : "https://www.xilinx.com/bin/public/openDownload?filename=densebox_320_320-u50lv-DPUCAHX8H-r2.5.0.tar.gz"
}

u200_u250 = {
  "tf_resnet_v1_50" : "https://www.xilinx.com/bin/public/openDownload?filename=resnet_v1_50_tf-u200-u250-r2.5.0.tar.gz",
  "densebox_320_320" : "https://www.xilinx.com/bin/public/openDownload?filename=densebox_320_320-u200-u250-r1.4.0.tar.gz"
}

zcu_102_104 = {
  "cf_resnet50" : "https://www.xilinx.com/bin/public/openDownload?filename=resnet50-zcu102_zcu104_kv260-r2.5.0.tar.gz"
}

# All artifacts
artifacts = {
  "u50lv_v3e" : u50lv_v3e,
  "u200_u250" : u200_u250,
  "zcu_102_104": zcu_102_104
}

# Argument list
parser = argparse.ArgumentParser(description='Download artifacts (xmodels)')
parser.add_argument('-f', '--force', action="store_true")
parser.add_argument('-d', '--device', action="store", type=str,
                    help="Device Name", required=False, default="all")

# Download artifacts
def main ():
  args = parser.parse_args()
  if (args.device == "all"):
    # Download all
    for akey, aval in artifacts.items():
      print ("Downloading artifacts for:", akey)
      epath = "graph_zoo/artifacts/" + akey
      if args.force:
        shutil.rmtree(epath)
        os.makedirs(epath)
      elif not os.path.exists(epath):
        os.makedirs(epath)
      else:
        continue
      for name, link in aval.items():
        print ("  Model:", name)
        dpath = "/tmp/" + akey + "_" + name + ".tar.gz"
        subprocess.call(["wget", link, "-O", dpath])
        subprocess.call(["tar", "-xzvf", dpath, "-C", epath])
        print()
  else:
    # Download for specific device
    print ("Downloading artifacts for:", args.device)
    epath = "graph_zoo/artifacts/" + args.device
    print ("  Extracting in:", epath)
    if args.force:
      shutil.rmtree(epath)
      os.makedirs(epath)
    elif not os.path.exists(epath):
      os.makedirs(epath)
    else:
      return
    for name, link in artifacts[args.device].items():
      print ("  Model:", name)
      dpath = "/tmp/" + args.device + "_" + name + ".tar.gz"
      subprocess.call(["wget", link, "-O", dpath])
      subprocess.call(["tar", "-xzvf", dpath, "-C", epath])
      print()

# Start
if __name__ == "__main__":
  main()
