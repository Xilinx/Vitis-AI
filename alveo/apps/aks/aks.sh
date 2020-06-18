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
  echo "    ./aks.sh --nfpga <number-of-fpgas> --impl <py/cpp>" 
  echo "             --model <model-variant>"
  echo "             --dir1 <image-dir> --dir2 <image-dir>"
  echo -e ""
  echo "Examples:"
  echo "------------------------------------------------"
  echo "Run GoogleNet with AKS C++: "
  echo "    ./aks.sh --impl cpp --model googlenet --dir1 ~/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min"
  echo -e ""
  echo "Run tinyyolov3 on video with AKS C++: "
  echo "    ./aks.sh --impl cpp --model tinyyolov3_video -vf ../../examples/vitis_ai_alveo_samples/video_analysis/video/structure.mp4"
  echo -e ""
  echo "Run ResNet50 with AKS Python: "
  echo "    ./aks.sh --impl py --model resnet50 --dir1 ~/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min"
  echo -e ""
  echo "Run Multinet example:"
  echo "    ./aks.sh --impl cpp --model googlenet_tinyyolov3"
  echo "    ./aks.sh --impl cpp --model googlenet_resenet50"
  echo -e ""
  echo "Run AKS C++ (Multiple FPGAs): "
  echo "    ./aks.sh --nfpga <N FPGAs on system> --impl cpp --model googlenet_resenet50"
  echo -e ""
  echo "Run GoogleNet with FPGA accelerated Pre-Processing: "
  echo "    ./aks.sh --impl cpp --model googlenet_pp_accel"

  echo -e ""
  echo "Arguments:"
  echo "------------------------------------------------"
  echo "  -n  nFPGA   | --nfpga  nFPGA    Number of FPGAs (Connected on System)"
  echo "  -m  MODEL   | --model  MODEL    Model variant"
  echo "                                  Possible values: [googlenet, resnet50]"
  echo "                                  Possible values: [tinyyolov3, stdyolov2]"
  echo "                                  Possible values: [googlenet_tinyyolov3]"
  echo "                                  Possible values: [googlenet_resnet50]"
  echo "                                  Possible values: [googlenet_pp_accel]"
  echo "                                  Possible values: [tinyyolov3_video]"
  echo "  -i  IMPL    | --impl   IMPL     Implemetation"
  echo "                                  Possible values: [cpp, py]"
  echo "  -d1 IMAGES  | --dir1   IMAGES   Classification Network's Image Directory"
  echo "  -d2 IMAGES  | --dir2   IMAGES   Detection Network's Image Directory"
  echo "  -vf VIDEO   | --video  VIDEO    Detection Network's Video File"
  echo "  -h          | --help            Print this message."
  echo "------------------------------------------------"
  echo -e ""
}

# Default
NUM_FPGA=""
MODEL="googlenet"
IMPL="cpp"
VERBOSE=1
C_DIRECTORY=~/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min
D_DIRECTORY=../yolo/test_image_set
D_VIDEO="../../examples/vitis_ai_alveo_samples/video_analysis/video/structure.mp4"

# Parse Options
while true
do
  if [[ -z "$1" ]]; then break; fi
  case "$1" in
    -n  |--nfpga         ) NUM_FPGA="$2"         ; shift 2 ;;
    -m  |--model         ) MODEL="$2"            ; shift 2 ;;
    -d1 |--dir1          ) C_DIRECTORY="$2"      ; shift 2 ;;
    -d2 |--dir2          ) D_DIRECTORY="$2"      ; shift 2 ;;
    -vf |--video         ) D_VIDEO="$2"          ; shift 2 ;;
    -i  |--impl          ) IMPL="$2"             ; shift 2 ;;
    -v  |--verbose       ) VERBOSE="$2"          ; shift 2 ;;
    -h  |--help          ) usage                 ; exit  1 ;;
     *) echo "Unknown argument : $1";
        echo "Try ./aks.sh -h to get correct usage. Exiting ...";
        exit 1 ;;
  esac
done

# Supported Modes & Models
declare -A SUPPORTED_MODELS
for name in "googlenet" "resnet50" "googlenet_resnet50" "tinyyolov3" "tinyyolov3_video" "googlenet_tinyyolov3" "stdyolov2" "facedetect" "googlenet_pp_accel"
do
    SUPPORTED_MODELS[$name]=1
done

if [[ ${SUPPORTED_MODELS["$MODEL"]} ]]; then 
  # Start Execution
  echo -e ""
  echo -e "[INFO] Running"
  echo -e "[INFO] Model: $MODEL with $IMPL"
  echo -e ""
else
  echo -e ""
  echo -e "[ERROR] $MODEL is an invalid model !"
  echo -e "[ERROR] Check Usage with: ./aks.sh -h "
  echo -e ""
  exit 1
fi

# Add verbose level
export AKS_VERBOSE=$VERBOSE

# Add AKS utils to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:${VAI_ALVEO_ROOT}

# Add Library Paths
if [ -d "${VAI_ALVEO_ROOT}/vai/dpuv1/rt/xdnn_cpp/lib" ]
then
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${VAI_ALVEO_ROOT}/vai/dpuv1/rt/xdnn_cpp/lib
fi
if [ -d "${VAI_ALVEO_ROOT}/vai/dpuv1/utils" ]
then
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${VAI_ALVEO_ROOT}/vai/dpuv1/utils
fi
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${VAI_ALVEO_ROOT}/apps/aks/libs
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib/python3.6/site-packages/vai/dpuv1/utils
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
export PYTHONPATH=${VAI_ALVEO_ROOT}/apps/face_detect:${VAI_ALVEO_ROOT}/apps/aks/libs:${VAI_ALVEO_ROOT}/apps/aks/libs/pykernels:${PYTHONPATH}

CARDS_CONNECTED=`xbutil scan | grep "xilinx_u" | wc -l`
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
    export NUM_FPGA=$NUM_FPGA
  fi
else
  unset NUM_FPGA
fi

CPP_EXE=""
PY_EXE=""
AKS_GRAPH_META_URL="https://www.xilinx.com/bin/public/openDownload?filename=aksMeta_vai1p2_16062020.zip"
# Check if the model files exists
for name in "meta_googlenet" "meta_resnet50" "meta_tinyyolov3" "meta_stdyolov2" "meta_googlenet_no_xbar" "meta_facedetect"
do
    NAME="graph_zoo/$name"
    if [[ ! -d "${NAME}" ]]; then
      echo -e "$NAME doesn't exist"
      wget $AKS_GRAPH_META_URL -O temp.zip && unzip temp.zip -d graph_zoo/ && rm temp.zip
      if [[ $? != 0 ]]; then echo "Network download failed. Exiting ..."; exit 1; fi;
      break;
    fi;
done

# Model Selection
if [ "$MODEL" == "googlenet" ]; then
  CPP_EXE=examples/bin/googlenet.exe
  PY_EXE=examples/googlenet.py
  exec_args="$C_DIRECTORY"

elif [ "$MODEL" == "googlenet_pp_accel" ]; then
  CPP_EXE=examples/bin/googlenet_pp_accel.exe
  PY_EXE=examples/googlenet_pp_accel.py
  exec_args="$C_DIRECTORY"

elif [ "$MODEL" == "resnet50" ]; then
  CPP_EXE=examples/bin/resnet50.exe
  PY_EXE=examples/resnet50.py
  exec_args="$C_DIRECTORY"

elif [ "$MODEL" == "tinyyolov3" ]; then
  CPP_EXE=examples/bin/tinyyolov3.exe
  PY_EXE=examples/tinyyolov3.py
  exec_args="$D_DIRECTORY"

elif [ "$MODEL" == "tinyyolov3_video" ]; then
  if [[ "$IMPL" == "py" ]]; then
    echo ""
    echo "[INFO] Model: tinyyolov3_video only has C++ implementation."
    echo ""
    exit 1
  fi
  CPP_EXE=examples/bin/tinyyolov3_video.exe
  exec_args="$D_VIDEO"

elif [ "$MODEL" == "stdyolov2" ]; then
  CPP_EXE=examples/bin/stdyolov2.exe
  PY_EXE=examples/stdyolov2.py
  exec_args="$D_DIRECTORY"

elif [ "$MODEL" == "googlenet_resnet50" ]; then
  CPP_EXE=examples/bin/googlenet_resnet50.exe
  PY_EXE=examples/googlenet_resnet50.py
  exec_args="$C_DIRECTORY"

elif [ "$MODEL" == "googlenet_tinyyolov3" ]; then
  CPP_EXE=examples/bin/googlenet_tinyyolov3.exe
  PY_EXE=examples/googlenet_tinyyolov3.py
  exec_args="$C_DIRECTORY $D_DIRECTORY"

elif [ "$MODEL" == "facedetect" ]; then
  echo "[INFO] Visit $VAI_ALVEO_ROOT/apps/face_detect to prepare FDDB dataset."
  echo -n "[INFO] To save results to a text file for accuracy measurement, "
  echo "provide a text file path to 'save_result_txt' argument in the graph_zoo/graph_facedetect.json"
  echo "[INFO] Visit $VAI_ALVEO_ROOT/apps/face_detect to see how to measure accuracy from the text file"
  echo ""
  CPP_EXE=examples/bin/facedetect.exe
  PY_EXE=examples/facedetect.py
  exec_args="$C_DIRECTORY"
fi

if [[ "$IMPL" == "cpp" ]]; then
  ${CPP_EXE} $exec_args
elif [[ "$IMPL" == "py" ]]; then
  python ${PY_EXE} $exec_args
fi
