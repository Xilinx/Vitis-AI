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

usage() {
  echo -e ""
  echo "Usage:"
  echo "------------------------------------------------"
  echo "  ./aks.sh --nfpga | -n  <number-of-fpgas>"
  echo "           --impl  | -i  <py/cpp>"
  echo "           --model | -m  <model-variant>"
  echo "           --dir1  | -d1 <image-dir>"
  echo "           --dir2  | -d2 <image-dir>"
  echo "           --video | -vf <video-file>"
  echo -e ""
  echo "Examples (DPUCADX8G):"
  echo "------------------------------------------------"
  echo "Run GoogleNet with AKS C++: "
  echo "    ./aks.sh -i cpp -m googlenet -d1 <image-dir>"
  echo -e ""
  echo "Run TinyYolov3 on video with AKS C++: "
  echo "    ./aks.sh -i cpp -m tinyyolov3_video -vf <video-file>"
  echo -e ""
  echo "Run ResNet50 with AKS Python: "
  echo "    ./aks.sh -i py -m resnet50 -d1 <image-dir>"
  echo -e ""
  echo "Run Multinet example:"
  echo "    ./aks.sh -i cpp --model googlenet_tinyyolov3 \\"
  echo "             -d1 <image-dir-for-googlenet> \\"
  echo "             -d2 <image-dir-for-tinyyolov3>"
  echo "    ./aks.sh -i cpp --model googlenet_resenet50 \\"
  echo "             -d1 <image-dir-for-googlenet> \\"
  echo "             -d2 <image-dir-for-resnet50>"
  echo -e ""
  echo "Run AKS C++ (Multiple FPGAs): "
  echo "    ./aks.sh -n <#FPGAs> -i cpp -m googlenet -d1 <image-dir>"
  echo -e ""
  echo "Run GoogleNet with FPGA accelerated Pre-Processing: "
  echo "    ./aks.sh -i cpp -m googlenet_pp_accel -d1 <image-dir>"
  echo -e ""

  echo "Examples (DPUCAHX8H):"
  echo "------------------------------------------------"
  echo "Run ResNet50 on Alveo-U50: "
  echo "    ./aks.sh -m resnet50_u50 --dir1 <image-dir>"
  echo -e ""

  echo "Examples (DPUCZDX8G):"
  echo "------------------------------------------------"
  echo "Run ResNet50 on Edge Platforms: "
  echo "    ./aks.sh -m resnet50_edge --dir1 <image-dir>"
  echo -e ""

  echo "Examples New DPU for Alveo-U200/U250 (DPUCADF8H):"
  echo "------------------------------------------------"
  echo "Run ResNet50 on Alveo-U200/U250: "
  echo "    ./aks.sh -m resnet50_cadf8h --dir1 <image-dir>"
  echo -e ""

  echo "Arguments:"
  echo "------------------------------------------------"
  echo "  -n  nFPGA   | --nfpga   nFPGA   Number of FPGAs (Connected on System)"
  echo "  -m  MODEL   | --model   MODEL   Model variant"
  echo "                                  Possible values: [googlenet, resnet50]"
  echo "                                  Possible values: [tinyyolov3, stdyolov2]"
  echo "                                  Possible values: [googlenet_tinyyolov3]"
  echo "                                  Possible values: [googlenet_resnet50]"
  echo "                                  Possible values: [googlenet_pp_accel]"
  echo "                                  Possible values: [tinyyolov3_video]"
  echo "                                  Possible values: [facedetect]"
  echo "                                  Possible values: [resnet50_edge] - only on edge devices"
  echo "                                  Possible values: [resnet50_u50] - only on U50 devices"
  echo "                                  Possible values: [resnet50_cadf8h] - only on U200/U250 devices"
  echo "  -i  IMPL    | --impl    IMPL    Implemetation"
  echo "                                  Possible values: [cpp, py]"
  echo "  -d1 IMAGES  | --dir1    IMAGES  Image Directory"
  echo "  -d2 IMAGES  | --dir2    IMAGES  Image Directory (To be used for Multi-Net)"
  echo "  -vf VIDEO   | --video   VIDEO   Video File"
  echo "  -v  VERBOSE | --verbose VERBOSE Verbosity level"
  echo "                                  Possible values: [0 - Only Warnings & Errors]"
  echo "                                  Possible values: [1 - Important Information, warnings & errors]"
  echo "                                  Possible values: [2 - All debug, performance metrics, warnings & errors]"
  echo "  -h          | --help            Print this message."
  echo "------------------------------------------------"
  echo -e ""
}

# Defaults
NUM_FPGA=""
MODEL="googlenet"
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
    -n  |--nfpga         ) NUM_FPGA="$2"         ; shift 2 ;;
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

# Supported Modes & Models
declare -A SUPPORTED_MODELS
for name in "googlenet" "resnet50" "inception_v1_tf" "googlenet_resnet50" "tinyyolov3" "tinyyolov3_video" "googlenet_tinyyolov3" "stdyolov2" "facedetect" "googlenet_pp_accel" "resnet50_edge" "resnet50_u50" "resnet50_cadf8h"
do
    SUPPORTED_MODELS[$name]=1
done

if [[ ${SUPPORTED_MODELS["$MODEL"]} ]]; then
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
PYTHONPATH=${PYTHONPATH}:${AKS_ROOT}/libs:${AKS_ROOT}/libs/pykernels:/usr/lib

# Add Library Paths (DPUCADX8G)
if [ -d "${VAI_HOME}/vai/dpuv1/rt/xdnn_cpp/lib" ]
then
  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${VAI_HOME}/vai/dpuv1/rt/xdnn_cpp/lib
fi
if [ -d "${VAI_HOME}/vai/dpuv1/utils" ]
then
  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${VAI_HOME}/vai/dpuv1/utils
fi
PYTHONPATH=${PYTHONPATH}:${VAI_HOME}/examples:${VAI_HOME}/examples/DPUCADX8G/face_detect

if [ ! -z "${CONDA_PREFIX}" ]; then
  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib/python3.6/site-packages/vai/dpuv1/utils
  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib
fi

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
export PYTHONPATH=${PYTHONPATH}

# Check for number of FPGA Devices
if [[ `uname -m` != "aarch64" ]]; then
  CARDS_CONNECTED=`/opt/xilinx/xrt/bin/xbutil scan | grep "xilinx_u" | wc -l`
  if [ ! -z "${NUM_FPGA}" ]
  then
    if [ ${NUM_FPGA} -gt ${CARDS_CONNECTED} ]
    then
      echo -e "[ERROR] Number of FPGAs mentioned are more than installed on System."
      echo -e "[ERROR] Cards Mentioned: ${NUM_FPGA}"
      echo -e "[ERROR] Cards Connected: ${CARDS_CONNECTED}"
      echo -e ""
      exit 1
    else
      # Number of FPGAs to be used
      export NUM_FPGA=${NUM_FPGA}
    fi
  else
    unset NUM_FPGA
  fi
fi

CPP_EXE=""
PY_EXE=""

# Check if the model files exists, download if not
get_dpucadx8g_artifacts () {
  AKS_GRAPH_META_URL="https://www.xilinx.com/bin/public/openDownload?filename=aksMeta_vai1p2_30062020.zip"
  if [[ `uname -m` != "aarch64" ]]; then
    for name in "meta_googlenet" "meta_inception_v1_tf" "meta_resnet50" "meta_tinyyolov3" "meta_stdyolov2" "meta_googlenet_no_xbar" "meta_facedetect"
    do
      NAME="graph_zoo/$name"
      if [[ ! -d "${NAME}" ]]; then
        echo -e "${NAME} doesn't exist"
        wget ${AKS_GRAPH_META_URL} -O temp.zip && unzip temp.zip -d graph_zoo/ && rm temp.zip
        if [[ $? != 0 ]]; then
          echo "Network download failed. Exiting ...";
          exit 1
        fi
        break
      fi
    done
  fi
}

# Check input image/video args
if [ "${MODEL}" == "tinyyolov3_video" ]; then
  if [[ "${VIDEO}" == "" ]]; then
    echo -e "[ERROR] No input video: \"-vf\" required\n";
    exit 1;
  fi
else
  if [[ "${DIRECTORY1}" == "" ]]; then
    echo -e "[ERROR] No input image directory: \"-d1\" required\n";
    exit 1;
  fi
  if [[ ! -d "${DIRECTORY1}" ]]; then
    echo -e "[ERROR] ${DIRECTORY1} doesn't exist\n"
    exit 1;
  fi
fi

# Model Selection
if [ "${MODEL}" == "resnet50_cadf8h" ]; then
  PY_EXE=examples/resnet50_dpucadf8h.py
  CPP_EXE=examples/bin/resnet50_dpucadf8h.exe
  exec_args="$DIRECTORY1"
  export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/dpuv3int8
  PYTHON=/usr/bin/python3

elif [ "${MODEL}" == "googlenet" ]; then
  get_dpucadx8g_artifacts
  CPP_EXE=examples/bin/googlenet.exe
  PY_EXE=examples/googlenet.py
  exec_args="$DIRECTORY1"

elif [ "${MODEL}" == "resnet50_u50" ]; then
  CPP_EXE=examples/bin/resnet50_u50.exe
  PY_EXE=examples/resnet50_u50.py
  PYTHON=/usr/bin/python3
  exec_args="$DIRECTORY1"

elif [ "${MODEL}" == "inception_v1_tf" ]; then
  get_dpucadx8g_artifacts
  CPP_EXE=examples/bin/inception_v1_tf.exe
  PY_EXE=examples/inception_v1_tf.py
  exec_args="$DIRECTORY1"

elif [ "${MODEL}" == "googlenet_pp_accel" ]; then
  get_dpucadx8g_artifacts
  CPP_EXE=examples/bin/googlenet_pp_accel.exe
  PY_EXE=examples/googlenet_pp_accel.py
  exec_args="$DIRECTORY1"

elif [ "${MODEL}" == "resnet50" ]; then
  get_dpucadx8g_artifacts
  CPP_EXE=examples/bin/resnet50.exe
  PY_EXE=examples/resnet50.py
  exec_args="$DIRECTORY1"

elif [ "${MODEL}" == "tinyyolov3" ]; then
  get_dpucadx8g_artifacts
  CPP_EXE=examples/bin/tinyyolov3.exe
  PY_EXE=examples/tinyyolov3.py
  exec_args="$DIRECTORY1"

elif [ "${MODEL}" == "tinyyolov3_video" ]; then
  if [[ "$IMPL" == "py" ]]; then
    echo ""
    echo "[INFO] Model: tinyyolov3_video only has C++ implementation."
    echo ""
    exit 1
  fi
  get_dpucadx8g_artifacts
  CPP_EXE=examples/bin/tinyyolov3_video.exe
  exec_args="$VIDEO"

elif [ "${MODEL}" == "stdyolov2" ]; then
  get_dpucadx8g_artifacts
  CPP_EXE=examples/bin/stdyolov2.exe
  PY_EXE=examples/stdyolov2.py
  exec_args="$DIRECTORY1"

elif [ "${MODEL}" == "googlenet_resnet50" ]; then
  get_dpucadx8g_artifacts
  CPP_EXE=examples/bin/googlenet_resnet50.exe
  PY_EXE=examples/googlenet_resnet50.py
  exec_args="$DIRECTORY1 $DIRECTORY2"
  if [[ "${DIRECTORY2}" == "" ]]; then
    echo -e "[ERROR] No input image directory: \"-d2\" required\n";
    exit 1;
  fi
  if [[ ! -d "${DIRECTORY2}" ]]; then
    echo -e "[ERROR] ${DIRECTORY2} doesn't exist\n"
    exit 1;
  fi;

elif [ "${MODEL}" == "googlenet_tinyyolov3" ]; then
  get_dpucadx8g_artifacts
  CPP_EXE=examples/bin/googlenet_tinyyolov3.exe
  PY_EXE=examples/googlenet_tinyyolov3.py
  exec_args="$DIRECTORY1 $DIRECTORY2"
  if [[ "${DIRECTORY2}" == "" ]]; then
    echo -e "[ERROR] No input image directory: \"-d2\" required\n";
    exit 1;
  fi
  if [[ ! -d "${DIRECTORY2}" ]]; then
    echo -e "[ERROR] ${DIRECTORY2} doesn't exist\n"
    exit 1;
  fi;

elif [ "${MODEL}" == "facedetect" ]; then
  echo "[INFO] Visit $VAI_HOME/examples/DPUCADX8G/face_detect to prepare FDDB dataset."
  echo -n "[INFO] To save results to a text file for accuracy measurement, "
  echo "provide a text file path to 'save_result_txt' argument in the graph_zoo/graph_facedetect.json"
  echo "[INFO] Visit $VAI_HOME/examples/DPUCADX8G/face_detect to see how to measure accuracy from the text file"
  echo ""
  get_dpucadx8g_artifacts
  CPP_EXE=examples/bin/facedetect.exe
  PY_EXE=examples/facedetect.py
  exec_args="$DIRECTORY1"

elif [ "${MODEL}" == "resnet50_edge" ]; then
  CPP_EXE=examples/bin/resnet50_edge.exe
  PY_EXE=examples/resnet50_edge.py
  exec_args="$DIRECTORY1"
	if [[ `uname -m` != "aarch64" ]]; then
	  echo "[ERROR] resnet50_edge will work only on edge devices with Aarch64 Processor and DPUCZDX8G IP."
	fi
fi

if [[ "$IMPL" == "cpp" ]]; then
  ${CPP_EXE} ${exec_args}
elif [[ "$IMPL" == "py" ]]; then
  ${PYTHON} ${PY_EXE} ${exec_args}
fi
