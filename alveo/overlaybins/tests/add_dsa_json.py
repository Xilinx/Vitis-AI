#!/usr/bin/env python3
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

#!/usr/bin/env python3
# Goal is to traverse all xclbin files and append DSA_VERSION to the matching .json file
# python tests/add_dsa_json.py
from __future__ import print_function
import os
import subprocess

def has_dsa_json(xclbinfname, verbose=False):
    import json
    print("Reading {0}".format(xclbinfname))

    with open(jsonfname) as f:
        data = json.load(f)

    valid_dsa_version = "DSA_VERSION" in data.keys()

    return True if valid_dsa_version else False


def add_dsa_json(jsonfname, verbose=False):
    print("Reading JSON from {0}".format(jsonfname))

    xclbinfname = str.join('.',jsonfname.split('.')[0:-1])

    print("Reading DSA from {0}".format(xclbinfname))

    # mkdir tmp dir
    import shutil
    shutil.rmtree("tmp", True)
    os.mkdir("tmp")

    os.chdir("tmp")
    # xclbinsplit into tmp dir
    os.system("/opt/xilinx/xrt/bin/xclbinsplit ."+xclbinfname)

    # read project/platform.vendor project/platform.boardid project
    import xml.etree.ElementTree
    e = xml.etree.ElementTree.parse('split-xclbin.xml').getroot()

    os.chdir("..")

    vendor = e.find('platform').get('vendor')
    boardid = e.find('platform').get('boardid')
    name = e.find('platform').get('name')
    major = e.find('platform').find('version').get('major')
    minor = e.find('platform').find('version').get('minor')

    DSA_VERSION = "{0}_{1}_{2}_{3}_{4}".format(vendor,boardid,name,major,minor)
    print("DSA_VERSION: {0}".format(DSA_VERSION))



# Write to output...
    import json
    print("Reading {0}".format(jsonfname))

    with open(jsonfname) as f:
        data = json.load(f)

    data['DSA_VERSION'] = DSA_VERSION
    with open(jsonfname,'w') as f:
        json.dump(data,f,separators=(',\n',':'))

    return True


def mapdir(dirname, excludes, prefix=" "):
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
                if add_dsa_json(objname):
                    valid_count += 1
            pass
        elif os.path.isdir(objname):
#            print(prefix + 'Dir: {}'.format(objname))
            (local_valid, local_total) = mapdir(objname, excludes, prefix+" ")
            valid_count += local_valid
            total_count += local_total
        elif os.path.isfile(objname) and os.path.islink(objname):
#            print(prefix+'SymLink: {}'.format(objname))
            pass
        else:
#            print(prefix+'Other: {}'.format(objname))
            pass
    return (valid_count, total_count)


if __name__ == "__main__":
    excludes = []
    (valid, total) = mapdir('.', excludes)
#    print("Valid files: {0}/{1}".format(valid,total))
#    if valid != total:
#        exit(1)
