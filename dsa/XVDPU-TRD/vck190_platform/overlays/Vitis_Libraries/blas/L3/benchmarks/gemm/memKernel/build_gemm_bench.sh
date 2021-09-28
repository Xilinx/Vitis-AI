#!/usr/bin/env bash

# Copyright 2019 Xilinx, Inc.
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


if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` path_to_config_info"
  exit 0
elif [ "$1" == "" ]; then
  echo "Usage: `basename $0` path_to_config_info"
  exit 0
else
  dataType=$(grep BLAS_dataType $1 | sed 's/^BLAS_dataType=//')
  numKernels=$(grep BLAS_numKernels $1 | sed 's/^BLAS_numKernels=//')
  make cleanall
  echo ================================================
  echo Now build benchmark with $dataType type
  echo ================================================
  make host BLAS_dataType=${dataType} BLAS_numKernels=${numKernels}
fi


