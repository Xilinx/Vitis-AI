## Copyright 2019 Xilinx Inc.
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

XRT_URL=""
XRM_URL=""
U200_URL=""
U250_URL=""
XRT_INSTALLER=""
U200_INSTALLER=""
U250_INSTALLER=""

XRM_INSTALLER=""

OVERLAYBINS_URL=""
OVERLAYBINS_INSTALLER=""

XPLUSML_OVERLAYBINS_URL=""
XPLUSML_OVERLAYBINS_INSTALLER=""
INSTALLER=""

# Note: Ubuntu 20.04 and CentOS/RHEL 8 are not supported by this script
#       However, they likely work with Vitis-AI

##############################
# Download XRT/DSA
##############################
if [[ $distroname == *"Ubuntu 16.04"* ]]; then
  echo "Ubuntu 16.04"
  XRT_URL="https://www.xilinx.com/bin/public/openDownload?filename=xrt_202020.2.8.726_16.04-amd64-xrt.deb"
  XRT_INSTALLER=/tmp/xrt.deb
  U200_URL="https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u200-xdma-201830.2-2580015_16.04.deb"
  U200_INSTALLER=/tmp/u200.deb
  U250_URL="https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u250-xdma-201830.2-2580015_16.04.deb"
  U250_INSTALLER=/tmp/u250.deb
elif [[ $distroname == *"Ubuntu 18.04"* ]]; then
  echo "Ubuntu 18.04"
  XRT_URL="https://www.xilinx.com/bin/public/openDownload?filename=xrt_202020.2.8.726_18.04-amd64-xrt.deb"
  XRT_INSTALLER=/tmp/xrt.deb
  U200_URL="https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u200-xdma-201830.2-2580015_18.04.deb"
  U200_INSTALLER=/tmp/u200.deb
  U250_URL="https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u250-xdma-201830.2-2580015_18.04.deb"
  U250_INSTALLER=/tmp/u250.deb
elif [[ $distroname == *"CentOS"* ]]; then
  echo "CentOS"
  XRT_URL="https://www.xilinx.com/bin/public/openDownload?filename=xrt_202010.2.6.655_7.4.1708-x86_64-xrt.rpm"
  XRT_INSTALLER=/tmp/xrt.rpm
  U200_URL="https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u200-xdma-201830.2-2580015.x86_64.rpm"
  U200_INSTALLER=/tmp/u200.rpm
  U250_URL="https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u250-xdma-201830.2-2580015.x86_64.rpm"
  U250_INSTALLER=/tmp/u250.rpm
elif [[ $distroname == *"Red Hat"* ]]; then
  echo "RHEL"
  XRT_URL="https://www.xilinx.com/bin/public/openDownload?filename=xrt_202020.2.8.726_7.4.1708-x86_64-xrt.rpm"
  XRT_INSTALLER=/tmp/xrt.rpm
  U200_URL="https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u200-xdma-201830.2-2580015.x86_64.rpm"
  U200_INSTALLER=/tmp/u200.rpm
  U250_URL="https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u250-xdma-201830.2-2580015.x86_64.rpm"
  U250_INSTALLER=/tmp/u250.rpm
else
  echo "Failed, couldn't detect os distribution"
  exit 1
fi


##############################
# Download XRM/Overlaybins
##############################
if [[ $distroname == *"Ubuntu 16.04"* || $distroname == *"Ubuntu 18.04"* ]]; then
  XRM_URL="https://www.xilinx.com/bin/public/openDownload?filename=xbutler_4.0-0.deb"
  XRM_INSTALLER=/tmp/xbutler_4.0-0.deb
  OVERLAYBINS_URL="https://www.xilinx.com/bin/public/openDownload?filename=vai-1.3-xilinx-overlaybins.deb"
  OVERLAYBINS_INSTALLER=/tmp/xilinx-overlaybins.deb
  XPLUSML_OVERLAYBINS_URL="https://www.xilinx.com/bin/public/openDownload?filename=vai-1.3-xplusml-bins-18.04.deb"
  XPLUSML_OVERLAYBINS_INSTALLER=/tmp/vai-1.3-xplusml-bins-18.04.deb
  INSTALLER="apt"
elif [[ $distroname == *"CentOS"* || $distroname == *"Red Hat"* ]]; then
  XRM_URL="https://www.xilinx.com/bin/public/openDownload?filename=xbutler-4.0.0-1.x86_64.rpm"
  XRM_INSTALLER=./centos/xbutler-3.0.2-1.el7.centos.x86_64.rpm
  OVERLAYBINS_URL="https://www.xilinx.com/bin/public/openDownload?filename=vai-1.3-xilinx-overlaybins.rpm"
  OVERLAYBINS_INSTALLER=/tmp/xilinx-overlaybins.rpm
  XPLUSML_OVERLAYBINS_URL="https://www.xilinx.com/bin/public/openDownload?filename=vai-1.3-xplusml-bins-1-0.x86_64.rpm"
  XPLUSML_OVERLAYBINS_INSTALLER=/tmp/vai-1.3-xplusml-bins-1.0-0.x86_64.rpm
  INSTALLER="yum"
else
  echo "Failed, couldn't detect os distribution"
  exit 1
fi

##############################
# Setup Install for CentOS and Red Hat
##############################
if [[ $distroname == *"CentOS"* ]]; then
  yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
elif [[ $distroname == *"Red Hat"* ]]; then
  yum-config-manager --enable rhel-7-server-optional-rpms
  yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
fi


##############################
# XILINX_XRT defined?
##############################
echo "----------------------"
echo "Verifying XILINX_XRT"
echo "----------------------"
if [ ! -f /opt/xilinx/xrt/setup.sh ]; then
	echo "XRT not found!"
	echo "Download and installing XRT..."
	wget $XRT_URL -O $XRT_INSTALLER && ${INSTALLER} install $XRT_INSTALLER -y && rm $XRT_INSTALLER
	echo "Download and installing XRT... done!"
else
	echo "XRT found!"
fi

##############################
# Install DSA/Overlaybins
##############################
echo "----------------------"
echo "Download and Install DSA/Overlaybins"
echo "----------------------"
wget $U200_URL -O $U200_INSTALLER && ${INSTALLER} install $U200_INSTALLER -y && rm $U200_INSTALLER
wget $U250_URL -O $U250_INSTALLER && ${INSTALLER} install $U250_INSTALLER -y && rm $U250_INSTALLER
#wget $OVERLAYBINS_URL -O $OVERLAYBINS_INSTALLER && ${INSTALLER} install $OVERLAYBINS_INSTALLER -y && rm $OVERLAYBINS_INSTALLER
#if [[ $distroname == *"Ubuntu 16.04"* || $distroname == *"Ubuntu 18.04"* ]]; then
#  wget $XPLUSML_OVERLAYBINS_URL -O $XPLUSML_OVERLAYBINS_INSTALLER && ${INSTALLER} install $XPLUSML_OVERLAYBINS_INSTALLER -y && rm $XPLUSML_OVERLAYBINS_INSTALLER
#elif [[ $distroname == *"CentOS"* || $distroname == *"Red Hat"* ]]; then
#  wget $XPLUSML_OVERLAYBINS_URL -O $XPLUSML_OVERLAYBINS_INSTALLER && rpm -ivh --force  $XPLUSML_OVERLAYBINS_INSTALLER && rm $XPLUSML_OVERLAYBINS_INSTALLER
#else
#  echo "Couldn't install WAA overlaybins. Unsupported operating system"
#fi

XCLBIN_URL="https://www.xilinx.com/bin/public/openDownload?filename=xdnnv3_xclbins_1_3_0.tar.gz"
XCLBIN_INSTALLER="/tmp/xclbins.tar.gz"

wget $XCLBIN_URL -O $XCLBIN_INSTALLER && tar -xzf $XCLBIN_INSTALLER --directory / && rm $XCLBIN_INSTALLER
sudo chmod -R a+r /opt/xilinx/overlaybins/xdnnv3

XCLBIN_URL="https://www.xilinx.com/bin/public/openDownload?filename=dpuv3int8_xclbins_1_3_0.tar.gz"
XCLBIN_INSTALLER="/tmp/xclbins.tar.gz"

wget $XCLBIN_URL -O $XCLBIN_INSTALLER && tar -xzf $XCLBIN_INSTALLER --directory / && rm $XCLBIN_INSTALLER
sudo chmod -R a+r /opt/xilinx/overlaybins/dpuv3int8

##############################
#TODO: detect datacenters
##############################
# install xrt_azure, xrt_aws, etc...


# reload drivers for XRT (xocl, xclmgmt)
modprobe -r xocl
modprobe -r xclmgmt
modprobe xocl
modprobe xclmgmt

##############################
# install XRM
##############################
echo "----------------------"
echo "Install XRM"
echo "----------------------"
wget $XRM_URL -O $XRM_INSTALLER && ${INSTALLER} install $XRM_INSTALLER -y && rm $XRM_INSTALLER

################
# Dead Code
# keep for 
# reference
################
# Detect Python
if [ 1 -eq 0 ]
then
	PYTHON_EXISTS=`which python`
	result=$?

	echo "----------------------"
	echo "Attemptimg to Flash Cards"
	echo "----------------------"
	# Python exists
	if [ "${result}" -eq "0" ]; then
		# detect cards
		CARDS=`python DSANameDetect.py`
		result=$?
		if [ -z "$CARDS" ]; then
			result=1
		fi
		if [ "${result}" -eq "0" ]; then
			# flash cards
			for card in ${CARDS}; do
				/opt/xilinx/xrt/bin/xbutil flash -a ${card}
			done
		fi
	fi
fi

# source XRT
echo "----------------------"
echo "Attemptimg to Flash Cards"
echo "----------------------"
source /opt/xilinx/xrt/setup.sh
SUPPORTED_DEVICES="u200 u250"
for device in ${SUPPORTED_DEVICES}
do
	/opt/xilinx/xrt/bin/xbutil flash -a xilinx_${device}_xdma_201830_2 <<< y
done
