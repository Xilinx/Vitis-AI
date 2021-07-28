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
if [[ $distroname == *"Ubuntu 16.04"* || $distroname == *"Ubuntu 18.04"* || $distroname == *"Ubuntu 20.04"* ]]; then
  XSA_URL="https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u200-gen3x16-xdma-all_1-3209015.deb_2.tar.gz"
  XSA_INSTALLER=/tmp/xsa.tar.gz
  INSTALLER="apt"
elif [[ $distroname == *"CentOS"* || $distroname == *"Red Hat"* ]]; then
  XSA_URL="https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u200-gen3x16-xdma-noarch_1-3209015.rpm_2.tar.gz"
  XSA_INSTALLER=/tmp/xsa.tar.gz
  INSTALLER="yum"
else
  echo "Failed, couldn't detect os distribution"
  exit 1
fi

XSA_DIR="/tmp/xsa"
mkdir $XSA_DIR

wget $XSA_URL -O $XSA_INSTALLER
tar -xzf $XSA_INSTALLER --directory $XSA_DIR
sudo $INSTALLER install $XSA_DIR/*cmc* -y
sudo $INSTALLER install $XSA_DIR/*sc-fw* -y
sudo $INSTALLER install $XSA_DIR/*validate* -y
sudo $INSTALLER reinstall $XSA_DIR/*base* -y
rm $XSA_INSTALLER
rm -rf $XSA_DIR

##############################
# Flash alveo
##############################
sudo /opt/xilinx/xrt/bin/xbmgmt flash --update --shell xilinx_u200_gen3x16_xdma_base_1
