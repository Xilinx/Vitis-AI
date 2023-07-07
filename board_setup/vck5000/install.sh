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

# Don't exit on error, do your best
set +e

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

##############################
# Setup Install for CentOS and Red Hat
##############################
if [[ $distroname == *"CentOS"* ]]; then
  sudo yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
elif [[ $distroname == *"Red Hat"* ]]; then
  sudo yum-config-manager --enable rhel-7-server-optional-rpms
  sudo yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
fi

SCRIPTS=./scripts

source ${SCRIPTS}/install_xrt.sh

##############################
# Install the required shells
##############################
# reload drivers for XRT (xocl, xclmgmt)

# This is sometimes helpful, sometimes troublesome
# For now it is being removed
#sudo modprobe -r xocl
#sudo modprobe -r xclmgmt
#sudo modprobe xocl
#sudo modprobe xclmgmt

sleep 3

source /opt/xilinx/xrt/setup.sh

echo "installing xclbins for vck5000"
platform=vck5000_
source ${SCRIPTS}/install_${platform}xclbins.sh

source ${SCRIPTS}/install_vck5000_shell.sh
source ${SCRIPTS}/install_xrm.sh
