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

XRM_URL=""
XRM_INSTALLER=""
INSTALLER=""

##############################
# Download XRM
##############################
if [[ $distroname == *"Ubuntu 16.04"* || $distroname == *"Ubuntu 18.04"* || $distroname == *"Ubuntu 20.04"* ]]; then
  XRM_URL="https://www.xilinx.com/bin/public/openDownload?filename=xbutler_4.0-0.deb"
  XRM_INSTALLER=/tmp/xrm.deb
  INSTALLER="apt"
elif [[ $distroname == *"CentOS"* || $distroname == *"Red Hat"* ]]; then
  XRM_URL="https://www.xilinx.com/bin/public/openDownload?filename=xbutler-4.0.0-1.x86_64.rpm"
  XRM_INSTALLER=/tmp/xrm.rpm
  INSTALLER="yum"
else
  echo "Failed, couldn't support os distribution"
  exit 1
fi
	
wget $XRM_URL -O $XRM_INSTALLER && sudo ${INSTALLER} install $XRM_INSTALLER -y && rm $XRM_INSTALLER
