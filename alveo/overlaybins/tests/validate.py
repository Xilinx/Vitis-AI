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

'''
Goal is to traverse all xclbin files and find matching json files
For every json file it must have
* XDNN_VERSION_MAJOR
* XDNN_VERSION_MINOR
* XDNN_BITWIDTH
* SDX_VERSION
* DSA_VERSION
* XDNN_NUM_KERNELS
'''

# Run from MLsuite/overlaybins
# python tests/validate.py

# If you really run python2
from __future__ import print_function

import os
import subprocess

def validate_json(jsonfname, verbose=False):
    import json
    print("Reading {0}".format(jsonfname))
    with open(jsonfname) as f:
        data = json.load(f)

    valid_xdnn_version = \
                       "XDNN_VERSION_MAJOR" in data.keys() and \
                       "XDNN_VERSION_MINOR" in data.keys() and \
                       "XDNN_VERSION" not in data.keys()

    valid_xdnn_config = \
                      "XDNN_BITWIDTH" in data.keys() and \
                      "XDNN_NUM_KERNELS" in data.keys()

    valid_sdx_version = \
          "SDX_VERSION" in data.keys() and \
          "DSA_VERSION" in data.keys()

    if verbose:
        if valid_xdnn_version:
            print("XDNN VERSION: {}.{}".format(data["XDNN_VERSION_MAJOR"], data["XDNN_VERSION_MINOR"]))
        if valid_xdnn_config:
            print("XDNN BITWIDTH: {}".format(data["XDNN_BITWIDTH"]))
            print("XDNN NUM KERNELS: {}".format(data["XDNN_NUM_KERNELS"]))

    if not valid_xdnn_version or not valid_xdnn_config or not valid_sdx_version:
      print("Valid XDNN Version: {0}".format(valid_xdnn_version))
      print("Valid XDNN Config:  {0}".format(valid_xdnn_config))
      print("Valid SDX Version:  {0}".format(valid_sdx_version))

    return True if valid_xdnn_version and valid_xdnn_config and valid_sdx_version else False


def fixdir(dirname, excludes, prefix=" "):
    objs = os.listdir(dirname)
    objs = list(set(objs) - set(excludes))

    valid_count = 0
    total_count = 0
    for o in objs:
        objname = dirname + '/' + o
        if os.path.isfile(objname) and not os.path.islink(objname):
            #print(prefix + 'File: {}'.format(objname))
            if objname.endswith(".json"):
                total_count += 1;
                if validate_json(objname):
                    valid_count += 1
            pass
        elif os.path.isdir(objname):
#            print(prefix + 'Dir: {}'.format(objname))
            (local_valid, local_total) = fixdir(objname, excludes, prefix+" ")
            valid_count += local_valid
            total_count += local_total
        elif os.path.isfile(objname) and os.path.islink(objname):
#            print(prefix+'SymLink: {}'.format(objname))
            pass
        else:
#            print(prefix+'Other: {}'.format(objname))
            pass
    return (valid_count, total_count)
excludes = []
(valid, total) = fixdir('.', excludes)
print("Valid files: {0}/{1}".format(valid,total))
if valid != total:
    exit(1)
