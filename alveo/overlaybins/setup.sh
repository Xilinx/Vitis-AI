#!/usr/bin/env bash
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

export VAI_ALVEO_ROOT="$( readlink -f "$( dirname "${BASH_SOURCE[0]}" )/.." )"

echo "------------------"
echo "Using VAI_ALVEO_ROOT"
echo "------------------"
echo $VAI_ALVEO_ROOT
echo ""

echo "---------------------"
echo "Using LD_LIBRARY_PATH"
echo "---------------------"
echo $LD_LIBRARY_PATH


LIBXDNN_PATH=${VAI_ALVEO_ROOT}/vai/dpuv1/rt/xdnn_cpp/lib/libxfdnn.so
if [ -f $LIBXDNN_PATH ]; then
  echo "---------------------"
  echo "Developer Flow"
  echo "---------------------"
  PYTHONPATH=${VAI_ALVEO_ROOT}:${VAI_ALVEO_ROOT}/apps/yolo:${VAI_ALVEO_ROOT}/apps/yolo/nms:${VAI_ALVEO_ROOT}/xfmlp/python:${PYTHONPATH}
  ln -s $VAI_ALVEO_ROOT/vai/dpuv1/tools/compile/bin/vai_c_tensorflow.py $CONDA_PREFIX/bin/vai_c_tensorflow
  ln -s $CONDA_PREFIX/bin/decent_q $CONDA_PREFIX/bin/vai_q_tensorflow
  MLSUITE_ROOT=$VAI_ALVEO_ROOT
  export MLSUITE_ROOT
else
  LIBXDNN_PATH=${CONDA_PREFIX}/lib/libxfdnn.so
fi
export LIBXDNN_PATH
export PYTHONPATH

echo "-------------------"
echo "Using LIBXDNN_PATH"
echo "-------------------"
echo $LIBXDNN_PATH
echo ""

echo "-------------------"
echo "PYTHONPATH"
echo "-------------------"
echo $PYTHONPATH
echo ""

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export XBLAS_NUM_PREP_THREADS=4

export XRT_INI_PATH=${VAI_ALVEO_ROOT}/overlaybins/xrt.ini

##############################
# Enable XILINX_XRT
##############################
echo "---------------------"
echo "Verifying XILINX_XRT"
echo "---------------------"
if [ -f /opt/xilinx/xrt/include/version.h ]; then
	info_xrt=$(cat /opt/xilinx/xrt/include/version.h | grep xrt_build_version\\[ | sed 's/[^0-9.]*//g')
	major=$(echo ${info_xrt} | cut -f1 -d ".")
	minor=$(echo ${info_xrt} | cut -f2 -d ".")
	major_gt=$(expr ${major} \> 2)
	major_eq=$(expr ${major} = 2)
	minor_=$(expr ${minor} \>= 2)
	# check version
	if [ ${major_eq} -eq "1" ]; then
		if [ ${minor_} -eq "0" ]; then
			echo "Invalid XRT Version!"
			exit -1
		fi
	elif [ ${major_gt} -eq "0" ]; then
		echo "Invalid XRT Version!"
		exit -1
	fi
	# enable XILINX_XRT
	source /opt/xilinx/xrt/setup.sh
	export XILINX_XRT=/opt/xilinx/xrt
else
	echo "Xilinx XRT not found on machine!"
	exit -1
fi

# Build NMS for YOLOv2 Demos
#make -C ${VAI_ALVEO_ROOT}/apps/yolo/nms

#export XBLAS_EMIT_PROFILING_INFO=1
