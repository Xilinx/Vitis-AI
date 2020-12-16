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

#!/bin/bash

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


##############################
# Download Gen3x4 Platform
##############################
if [[ $distroname == *"Ubuntu 16.04"* ]]; then
  echo "Ubuntu 16.04"
  DEPLOY_PLFM_URL="https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u280-xdma-201920.3-2789161_16.04.deb"
elif [[ $distroname == *"Ubuntu 18.04"* ]]; then
  echo "Ubuntu 18.04"
  DEPLOY_PLFM_URL="https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u280-xdma-201920.3-2789161_18.04.deb"
elif [[ $distroname == *"CentOS"* ]] || [[ $distroname == *"Red Hat"* ]]; then
  echo "CentOS/RHEL"
  DEPLOY_PLFM_URL="https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u280-xdma-201920.3-2789161.x86_64.rpm"
else
  echo "Failed, couldn't detect os distribution"
  exit 1
fi

##############################
# Install Gen3x4 Platform
##############################
mkdir ./temp
cd ./temp
if [[ $distroname == *"Ubuntu 16.04"* ]] || [[ $distroname == *"Ubuntu 18.04"* ]]; then
  wget $DEPLOY_PLFM_URL -O shell.deb
  sudo apt install ./shell.deb -y
elif [[ $distroname == *"CentOS"* ]] || [[ $distroname == *"RHEL"* ]]; then
  wget $DEPLOY_PLFM_URL -O shell.rpm
  sudo yum install ./shell.rpm -y
else
  echo "Failed, couldn't detect os distribution"
  exit 1
fi

cd ..

##############################
# Flash alveo
##############################
sudo /opt/xilinx/xrt/bin/xbmgmt flash --update --shell xilinx_u280_xdma_201920_3
