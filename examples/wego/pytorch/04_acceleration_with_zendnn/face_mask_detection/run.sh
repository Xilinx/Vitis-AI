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

#! /bin/bash

export XLNX_BUFFER_POOL=32


usage()
{
    echo "Usage: bash run.sh --zendnn [true|false]"
    exit 0
}

if [ "$#" -ne 2 ];then
    usage
fi

while [[ $# -gt 0 ]];do
    case $1 in
        --zendnn) ZENDNN="$2"; shift;;
	*) echo "Unknown option $1"; usage; exit 1 ;;
    esac
    shift
done

if [ "$ZENDNN" != true ] && [ "$ZENDNN" != false ]; then
   echo "Unknown option value: $ZENDNN"
   usage
   exit 1
fi

if [ "$ZENDNN" = true ];then
    python run.py --model_path ./model/Darknet_int.pt \
              --threads 8 \
              --enable_zen
else
   python run.py --model_path ./model/Darknet_int.pt \
              --threads 8
fi

unset XLNX_BUFFER_POOL
