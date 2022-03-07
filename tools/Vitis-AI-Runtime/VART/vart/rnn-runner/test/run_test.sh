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

function print_usage {
  echo "./cmake.sh [options]"
  echo "    --help                    show help"
  echo "    --model-dir[=DIR]         model directory"
  echo "    --device                  target device. {u25, u50[default]}"
  echo "    --num-iters               number of iterations to run. [default=4]"
  echo "    --num-runners             number of runners to use in multi-runner tests. [default = 4]"
  echo "    --build-dir[=DIR]         set customized build directory. [default = ${DEFAULT_BUILD_DIR}]"
  echo "    --tests-dir[=DIR]         test data directory. [default = ${DEFAULT_TEST_DIR}]"
  echo "    --mode                    specify execution mode. {func|latency|throughput|all}. [default = func]"
}

function file_exists {
  [[ -z $1 ]] && echo "$1 doesn't exist" && exit 1
  [[ ! -f $1 ]] && echo "$1 doesn't exist" && exit 1
}

function dir_exists {
  [[ ! -d $1 ]] && echo "$1 doesn't exist" && exit 1
}

NRUNNERS=4
DEFAULT_NITER=4
DEVICE="u50"
DEFAULT_BUILD_DIR="/home/vitis-ai-user/build/build.Ubuntu.18.04.x86_64.Release"
DEFAULT_TEST_DIR="./data"
DEFAULT_MODE="func"
PROFILE=0


# parse options
options=$(getopt -a -n 'parse-options' -o h \
  -l help,device:,build-dir:,num-runners:,num-iters:,model-dir:,tests-dir:,mode: \
  -- "$0" "$@")
  [ $? -eq 0 ] || {
    echo "Failed to parse arguments!"
    print_usage
      exit 1
    }
  eval set -- "$options"
  while true; do
    case "$1" in
      -h | --help) show_help=true; break;;
      --build-dir) shift; BUILD_DIR=$1;;
      --model-dir) shift; MODELS_DIR=$1;;
      --tests-dir) shift; TESTDATA_DIR=$1;;
      --mode) shift; MODE=$1;;
      --device)
        shift
        case "$1" in
          u50) DEVICE=u50;;
          u25) DEVICE=u25;;
          *) echo "Invalid device \"$1\"! try --help"; exit 1;;
        esac
        ;;
      --num-iters) shift; USER_NITER=$1;;
      --num-runners) shift; NRUNNERS=$1;;
      --) shift; break;;
    esac
    shift
  done

if [ ${show_help:=false} == true ]; then
  print_usage
  exit 0
fi

[[ -z ${MODE} ]] && MODE=${DEFAULT_MODE}
[[ -z ${BUILD_DIR} ]] && BUILD_DIR=${DEFAULT_BUILD_DIR}
[[ -z ${TESTDATA_DIR} ]] && TESTDATA_DIR=${DEFAULT_TEST_DIR}
[[ -z ${MODELS_DIR} ]] && echo "Model directory not provided. Use --model_dir=DIR option" && exit 1
dir_exists ${BUILD_DIR}
dir_exists ${MODELS_DIR}
dir_exists ${TESTDATA_DIR}

EXECUTABLE=${BUILD_DIR}/vart/rnn-runner/test_rnn_runner_${DEVICE}
file_exists ${EXECUTABLE}

MODELS=("lstm_customer_satisfaction" "lstm_sentiment_detection" "openie-new" "openie-new" "gru_sentiment_detection")
MODEL_NAMES=("satisfaction" "sentiment" "openie-36" "openie-59" "gru")
FRAMES=(25 500 36 59 500)
DATA_DIR=("satis" "sent" "oie/36" "oie/59" "gru")
NUM_RUNNERS=(1 ${NRUNNERS})

if [[ ${MODE} == "func" || ${MODE} == "all" ]]; then
  [[ -z $USER_NITER ]] && NITER=4 || NITER=$USER_NITER
  OUTPUT_FILE="rnn_func_report.log"
  rm $OUTPUT_FILE 2>/dev/null
  for i in ${!MODELS[@]}; do
    for nrunner in ${NUM_RUNNERS[@]}; do
      CUR_MODEL_DIR=${MODELS_DIR}/${MODELS[$i]}
      CUR_TESTDATA_DIR=${TESTDATA_DIR}/${DEVICE}/${DATA_DIR[$i]}
      dir_exists ${CUR_MODEL_DIR}
      dir_exists ${CUR_TESTDATA_DIR}
      echo
      echo "-----------------------------------------------------------------------"
      CMD="${EXECUTABLE} ${CUR_MODEL_DIR} ${CUR_TESTDATA_DIR} ${FRAMES[$i]} ${nrunner} $NITER"
      echo $CMD
      $CMD
      echo "-----------------------------------------------------------------------"
      echo
    done
  done
fi

# Latency Breakup
if [[ ${MODE} == "latency" || ${MODE} == "all" ]]; then
  PROFILE_DIR="/tmp/rnn_prof"
  mkdir -p ${PROFILE_DIR} && rm -rf ${PROFILE_DIR}/*
  [[ -z $USER_NITER ]] && NITER=5000 || NITER=$USER_NITER
  for i in ${!MODELS[@]}; do
    BATCH=3
    CUR_MODEL_DIR=${MODELS_DIR}/${MODELS[$i]}
    CUR_TESTDATA_DIR=${TESTDATA_DIR}/${DEVICE}/${DATA_DIR[$i]}
    dir_exists ${CUR_MODEL_DIR}
    dir_exists ${CUR_TESTDATA_DIR}
    echo
    echo "-----------------------------------------------------------------------"
    LOGPATH="${PROFILE_DIR}/profile_${DEVICE}_${MODEL_NAMES[$i]}_${BATCH}.lat"
    CMD="${EXECUTABLE} ${CUR_MODEL_DIR} ${CUR_TESTDATA_DIR} ${FRAMES[$i]} ${NUM_RUNNERS[0]} $NITER"
    echo "$CMD 2>${LOGPATH}"
    DEEPHI_PROFILING=1 $CMD 2>${LOGPATH}
    echo "-----------------------------------------------------------------------"
    echo
  done
  python profile.py --log_dir=${PROFILE_DIR} --output_file=rnn_latency_report.csv
  echo
  cat rnn_latency_report.csv
fi

# FPS calculation
if [[ ${MODE} == "throughput" || ${MODE} == "all" ]]; then
  [[ -z $USER_NITER ]] && NITER=5000 || NITER=$USER_NITER
  [[ ${DEVICE} == "u50" ]] && NUMCU=2 || NUMCU=1
  NRUNNERS=(1 2 4 8)
  NEW_NITER=($NITER)
  for i in ${NRUNNERS[@]:1}; do
    NEW_NITER+=($((NITER * NUMCU / i)))
  done

  LOGPATH="/tmp/rnn_throughput.bps"
  OUTPUT_FILE="rnn_throughput_report.csv"
  [[ -f $OUTPUT_FILE ]] && rm $OUTPUT_FILE 2>/dev/null
  print_data=("model")
  for i in ${NRUNNERS[*]}; do
    print_data+=(", #Runners=$i")
  done
  echo ${print_data[*]} >> $OUTPUT_FILE

  for i in ${!MODELS[@]}; do
    print_data=(${MODELS[$i]}/${FRAMES[$i]})
    for j in ${!NRUNNERS[@]}; do
      rm $LOGPATH 2>/dev/null
      nrunner=${NRUNNERS[$j]}
      new_iter=${NEW_NITER[$j]}
      nframes=${FRAMES[$i]}
      CUR_MODEL_DIR=${MODELS_DIR}/${MODELS[$i]}
      CUR_TESTDATA_DIR=${TESTDATA_DIR}/${DEVICE}/${DATA_DIR[$i]}
      dir_exists ${CUR_MODEL_DIR}
      dir_exists ${CUR_TESTDATA_DIR}
      echo
      echo "-----------------------------------------------------------------------"
      CMD="${EXECUTABLE} ${CUR_MODEL_DIR} ${CUR_TESTDATA_DIR} ${nframes} ${nrunner} ${new_iter}"
      echo "$CMD 2>${LOGPATH}"
      $CMD 2>${LOGPATH}
      echo "-----------------------------------------------------------------------"
      echo
      [[ -f ${LOGPATH} ]] && bps=$(grep "Throughput" ${LOGPATH} | awk '{print $NF}') || bps=0
      print_data+=(", $bps")
    done
    echo ${print_data[*]} >> $OUTPUT_FILE
  done
  cat $OUTPUT_FILE
fi

