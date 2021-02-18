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
# Get XRT pack link
##############################
if [[ $distroname == *"Ubuntu 16.04"* ]]; then
  echo "Ubuntu 16.04"
  XRT_URL="https://www.xilinx.com/bin/public/openDownload?filename=xrt_202020.2.8.726_16.04-amd64-xrt.deb"
elif [[ $distroname == *"Ubuntu 18.04"* ]]; then
  echo "Ubuntu 18.04"
  XRT_URL="https://www.xilinx.com/bin/public/openDownload?filename=xrt_202020.2.8.726_18.04-amd64-xrt.deb"
elif [[ $distroname == *"Ubuntu 20.04"* ]]; then
  echo "Ubuntu 20.04"
  XRT_URL="https://www.xilinx.com/bin/public/openDownload?filename=xrt_202020.2.8.726_20.04-amd64-xrt.deb"
elif [[ $distroname == *"CentOS"* ]] || [[ $distroname == *"Red Hat"* ]]; then
  echo "CentOS/RHEL"
  XRT_URL="https://www.xilinx.com/bin/public/openDownload?filename=xrt_202020.2.8.726_7.4.1708-x86_64-xrt.rpm"
else
  echo "Failed, couldn't detect os distribution"
  exit 1
fi

##############################
# Install XRT Platform
##############################
mkdir ./temp
cd ./temp
if [[ $distroname == *"Ubuntu 16.04"* ]] || [[ $distroname == *"Ubuntu 18.04"* ]] || [[ $distroname == *"Ubuntu 20.04"* ]]; then
  wget $XRT_URL -O xrt.deb
  sudo apt install ./xrt*.deb -y
elif [[ $distroname == *"CentOS"* ]] || [[ $distroname == *"RHEL"* ]]; then
  sudo yum-config-manager --enable rhel-7-server-optional-rpms
  sudo yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
  wget $XRT_URL -O xrt.rpm
  sudo yum install ./xrt*.rpm -y
else
  echo "Failed, couldn't detect os distribution"
  exit 1
fi

cd ..

##############################
# Call platform install scripts
##############################
/opt/xilinx/xrt/bin/xbutil scan | grep xilinx_u50_
if [ $? -eq 0 ]; then
  echo "U50 card detected, now install platform"
  source ./u50_shell_setup.sh
fi

/opt/xilinx/xrt/bin/xbutil scan | grep xilinx_u50lv_
if [ $? -eq 0 ]; then
  echo "U50LV card detected, now install platform"
  source ./u50lv_shell_setup.sh
fi

/opt/xilinx/xrt/bin/xbutil scan | grep xilinx_u280_
if [ $? -eq 0 ]; then
  echo "U280 card detected, now install platform"
  source ./u280_shell_setup.sh
fi

##############################
# Downloads Overlays
##############################
wget https://www.xilinx.com/bin/public/openDownload?filename=alveo_xclbin-1.3.1.tar.gz -O alveo_xclbin-1.3.1.tar.gz
tar xfz alveo_xclbin-1.3.1.tar.gz
rm alveo_xclbin-1.3.1.tar.gz


echo "You may need to cold reboot the machine, please refer to the prompt above."
