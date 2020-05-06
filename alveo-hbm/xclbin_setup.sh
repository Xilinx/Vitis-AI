#!/bin/bash
# Copyright 2020 Xilinx Inc.
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

XCLBIN_URL='https://www.xilinx.com/bin/public/openDownload?filename=U50_gen3x4_xclbin_update1.tar.gz'
mkdir temp
cd temp
wget  $XCLBIN_URL -O ./U50_gen3x4_xclbin_update1.tgz
tar xfzv ./U50_gen3x4_xclbin_update1.tgz
sudo cp -p ./U50_gen3x4_xclbin_update1/6E275M/* /usr/lib
cd ..
rm -rf temp

