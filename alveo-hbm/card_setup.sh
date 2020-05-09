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

PLFM_G3X4_URL="https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u50-gen3x4-xdma-201920.3-2784799.noarch.rpm"
PLFM_G3X16_URL="https://www.xilinx.com/bin/public/openDownload?filename=Xilinx_u50-gen3x16-xdma-201920.3-2784799_noarch_rpm.tar.gz"


mkdir temp
cd temp


wget $PLFM_G3X16_URL -O installer.tgz
tar xfzv installer.tgz
sudo yum install -y ./xilinx-sc-fw-u50-5.0.27-2.e289be9.noarch.rpm
sudo yum install -y ./xilinx-cmc-u50-1.0.17-2784148.noarch.rpm

wget $PLFM_G3X4_URL -O installer.rpm
sudo yum install -y ./installer.rpm

cd ..
rm -rf temp

sudo xbmgmt flash --update --shell xilinx_u50_gen3x4_xdma_201920_3
