## Copyright 2020 Xilinx Inc.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

#!/bin/bash
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

arch=$(uname -m)
kernel=$(uname -r)
if [ -n "$(command -v lsb_release)" ]; then
  distroname=$(lsb_release -s -d)
elif [ -f "/etc/os-release" ]; then
  distroname=$(grep PRETTY_NAME /etc/os-release | sed 's/PRETTY_NAME=//g' | tr -d '="')
elif [ -f "/etc/debian_version" ]; then
  distroname="Debian $(cat /etc/debian_version)"
elif [ -f "/etc/redhat-release" ]; then
  distroname=$(cat /etc/redhat-release)
else
  distroname="$(uname -s) $(uname -r)"
fi

DEPLOY_PLFM_URL="https://www.xilinx.com/bin/public/openDownload?filename=dpuv3e_u50_platform_vai1.2.tgz"
wget $DEPLOY_PLFM_URL -O shell.tgz

##############################
# Install Gen3x4 Platform
##############################
tar xfz shell.tgz
cd ./dpuv3e_u50_platform_vai1.2

if [[ $distroname == *"Ubuntu 16.04"* ]]; then
  sudo apt install ./*cmc*16.04.deb -y --allow-downgrades
  sudo apt install ./*sc-fw*16.04.deb -y --allow-downgrades
  sudo apt install ./*-u50-*validate*16.04.deb -y --allow-downgrades
  sudo apt install ./*-u50-*base*16.04.deb -y --allow-downgrades
elif [[ $distroname == *"Ubuntu 18.04"* ]]; then
  sudo apt install ./*cmc*18.04.deb -y --allow-downgrades
  sudo apt install ./*sc-fw*18.04.deb -y --allow-downgrades
  sudo apt install ./*-u50-*validate*18.04.deb -y --allow-downgrades
  sudo apt install ./*-u50-*base*18.04.deb -y --allow-downgrades
elif [[ $distroname == *"CentOS"* ]] || [[ $distroname == *"RHEL"* ]]; then
  sudo yum install ./*cmc*.rpm -y
  sudo apt install ./*sc-fw*.rpm -y
  sudo apt install ./*-u50-*validate*.rpm -y
  sudo apt install ./*-u50-*base*.rpm -y
fi

sudo /opt/xilinx/xrt/bin/xbmgmt flash --update --shell xilinx_u50_gen3x4_xdma_base_2
