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
import os, argparse

homedir = os.environ['HOME']
mlsroot = os.environ['VAI_ALVEO_ROOT']

def substitute(filepath, replacee, replacement):
    # Read in the file
    with open(filepath, 'r') as f :
        filedata = f.read()
    # Replace the target string
    filedata = filedata.replace(replacee, replacement)
    # Write the file out again
    with open(filepath, 'w') as f:
        f.write(filedata)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replace CK-TOOLS Root paths in prototxt(s)')
    parser.add_argument('--rootdir', type=str, default=os.environ['HOME'], 
            help='Root path to CK-TOOLS directory. if not provided, "/home/mluser" gets replaced by "/home/username"')
    parser.add_argument('--modelsdir', type=str, default=mlsroot+"/models/container/caffe", 
            help='Root path to MODELS directory.')
    args = parser.parse_args()

    for Dir, subDirs, files in os.walk(args.modelsdir):
        success = [substitute(os.path.join(Dir, f), "/home/mluser", args.rootdir) for f in files if f.endswith(".prototxt")]
        success = [substitute(os.path.join(Dir, f), "/opt/ml-suite", mlsroot) for f in files if f.endswith(".prototxt")]
