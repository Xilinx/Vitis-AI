#!/bin/bash

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

TestBoard="$1"

[ -d "./model" ] && rm -rf "./model"

if [ "$TestBoard" = "zcu102" ] || [ "$TestBoard" = "ZCU102" ]; then
	if [ -e ./model_for_zcu102 ]; then
		echo "copy zcu102 model file..."
		cp -r ./model_for_zcu102/libdpumodeltf_yolov3_voc.so ./
	else
		echo "The folder named 'Model_for_zcu102' does not exist!" 
		exit 1
	fi
elif [ "$TestBoard" = "zcu104" ] || [ "$TestBoard" = "ZCU104" ]; then
	if [ -e ./model_for_zcu104 ]; then
		echo "copy zcu104 model file..."
		cp -r ./model_for_zcu104/libdpumodeltf_yolov3_voc.so ./
	else
		echo "The folder named 'Model_for_zcu104' does not exist!"
		exit 1
	fi
else 
	echo "Please enter the correct command: './build.sh zcu102' or './build.sh zcu104'!"
	exit 1
fi

