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

# Detect OS Distribution
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

XRT_URL=""
XRT_INSTALLER=""
INSTALLER=""

##############################
# Download XRT/DSA
##############################
if [[ $distroname == *"Ubuntu 18.04"* ]]; then
  XRT_URL="https://www.xilinx.com/bin/public/openDownload?filename=xrt_202220.2.14.354_18.04-amd64-xrt.deb"
  XRT_INSTALLER=/tmp/xrt.deb
  INSTALLER="apt"
elif [[ $distroname == *"Ubuntu 20.04"* ]]; then
  XRT_URL="https://www.xilinx.com/bin/public/openDownload?filename=xrt_202220.2.14.354_20.04-amd64-xrt.deb"
  XRT_INSTALLER=/tmp/xrt.deb
  INSTALLER="apt"
elif [[ $distroname == *"Ubuntu 22.04"* ]]; then
  XRT_URL="https://www.xilinx.com/bin/public/openDownload?filename=xrt_202220.2.14.354_22.04-amd64-xrt.deb"
  XRT_INSTALLER=/tmp/xrt.deb
  INSTALLER="apt"
elif [[ ( $distroname == *"CentOS"* || $distroname == *"Red Hat"* ) && $distroname == *"7.8"* ]]; then
  XRT_URL="https://www.xilinx.com/bin/public/openDownload?filename=xrt_202220.2.14.354_7.8.2003-x86_64-xrt.rpm"
  XRT_INSTALLER=/tmp/xrt.rpm
  INSTALLER="yum"
elif [[ ( $distroname == *"CentOS"* || $distroname == *"Red Hat"* ) && $distroname == *"7.9"* ]]; then
  XRT_URL="https://www.xilinx.com/bin/public/openDownload?filename=xrt_202220.2.14.354_7.9.2009-x86_64-xrt.rpm"
  XRT_INSTALLER=/tmp/xrt.rpm
  INSTALLER="yum"
elif [[ ( $distroname == *"CentOS"* || $distroname == *"Red Hat"* ) && $distroname == *"8.1"* ]]; then
  XRT_URL="https://www.xilinx.com/bin/public/openDownload?filename=xrt_202220.2.14.354_8.1.1911-x86_64-xrt.rpm"
  XRT_INSTALLER=/tmp/xrt.rpm
  INSTALLER="yum"
elif [[ ( $distroname == *"CentOS"* || $distroname == *"Red Hat"* ) && $distroname == *"8.2"* ]]; then
  XRT_URL="https://www.xilinx.com/bin/public/openDownload?filename=xrt_202220.2.14.354_8.1.1911-x86_64-xrt.rpm"
  XRT_INSTALLER=/tmp/xrt.rpm
  INSTALLER="yum"
elif [[ $distroname == *"Red Hat Enterprise Linux"* && $distroname == *"8.3"* ]]; then
  XRT_URL="https://www.xilinx.com/bin/public/openDownload?filename=xrt_202220.2.14.354_8.1.1911-x86_64-xrt.rpm"
  XRT_INSTALLER=/tmp/xrt.rpm
  INSTALLER="yum"
elif [[ $distroname == *"Red Hat Enterprise Linux"* && $distroname == *"8.4"* ]]; then
  XRT_URL="https://www.xilinx.com/bin/public/openDownload?filename=xrt_202220.2.14.354_8.1.1911-x86_64-xrt.rpm"
  XRT_INSTALLER=/tmp/xrt.rpm
  INSTALLER="yum"
else
  echo "Error: XRT does not support this OS"
  exit 1
fi
	
wget $XRT_URL -O $XRT_INSTALLER && sudo ${INSTALLER} install $XRT_INSTALLER -y && rm $XRT_INSTALLER
