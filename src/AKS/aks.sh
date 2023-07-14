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

# Supported Modes & Models
declare -A SUPPORTED_GRAPHS
for name in "cf_resnet50_zcu_102_104" "tf_resnet_v1_50_u200_u250" "tf_resnet_v1_50_u50lv_v3e"  \
            "cf_densebox_320_320_u200_u250" "cf_densebox_320_320_u50lv_v3e"
do
    SUPPORTED_GRAPHS[$name]=1
done

usage() {
  echo -e ""
  echo "Usage:"
  echo "------------------------------------------------"
  echo "  ./aks.sh --model | -m  <model-variant>"
  echo "           --dir1  | -d1 <image-dir>"
  echo "           --dir2  | -d2 <image-dir>"
  echo "           --video | -vf <video-file>"
  echo -e ""

  echo "Examples (DPUCAHX8H):"
  echo "------------------------------------------------"
  echo "Run ResNet50 on Alveo-U50LV: "
  echo "  ./aks.sh -m tf_resnet50_v1_50_u50lv_v3e --dir1 <image-dir>"
  echo "Run Face Detect on Alveo-U50: "
  echo "  ./aks.sh -m cf_densebox_320_320_u50lv_v3e --dir1 <image-dir>"
  echo -e ""

  echo "Examples (DPUCZDX8G):"
  echo "------------------------------------------------"
  echo "Run ResNet50 on Edge Platforms: "
  echo "  ./aks.sh -m cf_resnet50_zcu102_104 --dir1 <image-dir>"
  echo -e ""

  echo "Examples (DPUCADF8H):"
  echo "------------------------------------------------"
  echo "Run ResNet50 on Alveo-U200/U250: "
  echo "  ./aks.sh -m tf_resnet_v1_50_u200_u250 --dir1 <image-dir>"
  echo "Run Face Detect on Alveo-U200/U250: "
  echo "  ./aks.sh -m cf_densebox_320_320_u200_u250 --dir1 <image-dir>"
  echo -e ""

  echo "Possible Graphs: "
  for name in "${!SUPPORTED_GRAPHS[@]}"; do
    echo -e "  ${name}"
  done
  echo "------------------------------------------------"
  echo -e ""
}

# Defaults
MODEL=""
DIRECTORY1=
DIRECTORY2=
VIDEO=""
IMPL="cpp"
VERBOSE=1
PYTHON=python3

# Parse Options
while true
do
  if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    usage;
    exit 0;
  fi
  if [ -z "$1" ]; then
    break;
  fi
  if [ -z "$2" ]; then
    echo -e "\n[ERROR] Missing argument value for $1 \n";
    exit 1;
  fi
  case "$1" in
    -m  |--model         ) MODEL="$2"            ; shift 2 ;;
    -d1 |--dir1          ) DIRECTORY1="$2"       ; shift 2 ;;
    -d2 |--dir2          ) DIRECTORY2="$2"       ; shift 2 ;;
    -vf |--video         ) VIDEO="$2"            ; shift 2 ;;
    -i  |--impl          ) IMPL="$2"             ; shift 2 ;;
    -v  |--verbose       ) VERBOSE="$2"          ; shift 2 ;;
     *) echo "Unknown argument : $1";
        echo "Try ./aks.sh -h to get correct usage. Exiting ...";
        exit 1 ;;
  esac
done


if [[ "$MODEL" =  "" ]]; then
  echo -e ""
  echo -e "[ERROR] No graph selected !"
  echo -e "[ERROR] Check Usage with: ./aks.sh -h "
  echo -e ""
  exit 1
elif [[ ${SUPPORTED_GRAPHS["$MODEL"]} ]]; then
  # Start Execution
  echo -e ""
  echo -e "[INFO] Running"
  echo -e "[INFO] Model: ${MODEL} with $IMPL"
  echo -e ""
else
  echo -e ""
  echo -e "[ERROR] ${MODEL} is an invalid model !"
  echo -e "[ERROR] Check Usage with: ./aks.sh -h "
  echo -e ""
  exit 1
fi

# AKS Root Dir
export AKS_ROOT=$(pwd)
# Add verbose level
export AKS_VERBOSE=${VERBOSE}

# Update lib paths
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${AKS_ROOT}/libs
PYTHONPATH=${PYTHONPATH}:${AKS_ROOT}/libs:/usr/lib

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
export PYTHONPATH=${PYTHONPATH}

CPP_EXE=""
PY_EXE=""

# Model Selection
if [ "${MODEL}" == "tf_resnet_v1_50_u200_u250" ]; then
  CPP_EXE=examples/bin/tf_resnet_v1_50.exe
  exec_args="u200_u250 ${DIRECTORY1}"

elif [ "${MODEL}" == "tf_resnet_v1_50_u50lv_v3e" ]; then
  CPP_EXE=examples/bin/tf_resnet_v1_50.exe
  exec_args="u50lv_v3e ${DIRECTORY1}"

elif [ "${MODEL}" == "cf_resnet50_zcu_102_104" ]; then
  CPP_EXE=examples/bin/cf_resnet50.exe
  exec_args="zcu_102_104 ${DIRECTORY1}"

elif [ "${MODEL}" == "cf_densebox_320_320_u200_u250" ]; then
  CPP_EXE=examples/bin/cf_densebox_320_320.exe
  exec_args="u200_u250 $DIRECTORY1"

elif [ "${MODEL}" == "cf_densebox_320_320_u50lv_v3e" ]; then
  CPP_EXE=examples/bin/cf_densebox_320_320.exe
  exec_args="u50lv_v3e $DIRECTORY1"

elif [ "${MODEL}" == "cf_densebox_320_320_v3e" ]; then
  CPP_EXE=examples/bin/cf_densebox_320_320.exe
  exec_args="u50lv_v3e $DIRECTORY1"
fi

# Base/Root path for xmodel directories
export AKS_XMODEL_ROOT=${AKS_ROOT}/graph_zoo/

# Run target
if [[ "$IMPL" == "cpp" ]]; then
  ${CPP_EXE} ${exec_args}
elif [[ "$IMPL" == "py" ]]; then
  ${PYTHON} ${PY_EXE} ${exec_args}
fi
