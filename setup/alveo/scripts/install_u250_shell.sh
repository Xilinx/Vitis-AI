#!/usr/bin/env bash
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

XSA_URL=""
XSA_INSTALLER=""
INSTALLER=""

##############################
# Download DSA
##############################
if [[ $distroname == *"Ubuntu 16.04"* ]]; then
  XSA_URL="https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u250-xdma-201830.2-2580015_16.04.deb"
  XSA_INSTALLER=/tmp/xsa.deb
  INSTALLER="apt"
elif [[ $distroname == *"Ubuntu 18.04"* ]]; then
  XSA_URL="https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u250-xdma-201830.2-2580015_18.04.deb"
  XSA_INSTALLER=/tmp/xsa.deb
  INSTALLER="apt"
elif [[ $distroname == *"Ubuntu 20.04"* ]]; then
  XSA_URL="https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u250-xdma-201830.2-2580015_20.04.deb" # Doesn't exist yet
  XSA_INSTALLER=/tmp/xsa.deb
  INSTALLER="apt"
elif [[ $distroname == *"CentOS"* ]]; then
  XSA_URL="https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u250-xdma-201830.2-2580015.x86_64.rpm"
  XSA_INSTALLER=/tmp/xsa.rpm
  INSTALLER="yum"
elif [[ $distroname == *"Red Hat"* ]]; then
  XSA_URL="https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u250-xdma-201830.2-2580015.x86_64.rpm"
  XSA_INSTALLER=/tmp/xsa.rpm
  INSTALLER="yum"
else
  echo "Failed, couldn't detect os distribution"
  exit 1
fi

wget $XSA_URL -O $XSA_INSTALLER && sudo ${INSTALLER} install $XSA_INSTALLER -y && rm $XSA_INSTALLER

##############################
# Flash alveo
##############################
sudo /opt/xilinx/xrt/bin/xbmgmt flash --update --shell xilinx_u250_xdma_201830_2
