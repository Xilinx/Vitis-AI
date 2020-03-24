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
  echo "Usage:"
  echo "./run.sh --test <test> --model <model>"
  echo "./run.sh  -t    <test>  -m <model>"
  echo "<test>     : test_classify / streaming_classify / streaming_classify_fpgaonly"
  echo "<compilerOpt> : autoAllOpt / latency / throughput"
  echo "Some tests require a directory of images to process."
  echo "To process a directory in a non-standard location use -d <directory> or --directory <directory>"
  echo "Some tests require a batchSize argument to know how many images to load simultaneously."
  echo "To provide batchSize use --batchsize <batchsize>"
  echo "-c allows to choose compiler optimization, for example, latency or throughput or autoAllOptimizations."
  echo "-g runs entire test providing top-1 and top-5 results"

}

# Default
TEST="test_classify"
MODEL="googlenet_v1"
ACCELERATOR="0"
BATCHSIZE=4
VERBOSE=0
PERPETUAL=0
PROFILING_ENABLE=0
IMG_INPUT_SCALE=1.0
# These variables are used in case there multiple FPGAs running in parallel
NUMDEVICES=1
NUMSTREAMS=8
DEVICEID=0
NUMPREPPROC=4
COMPILEROPT="autoAllOpt"
# Parse Options
OPTS=`getopt -o t:m:d:s:a:n:ns:i:c:is:y:gvxph --long test:,model:,directory:,numdevices:,numstreams:,deviceid:,batchsize:,compilerOpt:,imagescale:,numprepproc,checkaccuracy,verbose,perpetual,profile,help -n "$0" -- "$@"`

if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; usage; exit 1 ; fi

while true
do
  case "$1" in
    -t |--test          ) TEST="$2"                                                       ; shift 2 ;;
    -m |--model         ) MODEL="$2"                                                      ; shift 2 ;;
    -d |--directory     ) DIRECTORY="$2"                                                  ; shift 2 ;;
    -r |--vitisrundir   ) VITIS_RUNDIR="$2"                                               ; shift 2 ;;
    -s |--batchsize     ) BATCHSIZE="$2"                                                  ; shift 2 ;;
    -a |--accelerator   ) ACCELERATOR="$2"                                                ; shift 2 ;;
    -n |--numdevices    ) NUMDEVICES="$2"                                                 ; shift 2 ;;
    -ns|--numstreams    ) NUMSTREAMS="$2"                                                 ; shift 2 ;;
    -i |--deviceid      ) DEVICEID="$2"                                                   ; shift 2 ;;
    -c |--compilerOpt   ) COMPILEROPT="$2"                                                ; shift 2 ;;
    -is|--imagescale    ) IMG_INPUT_SCALE="$2"                                            ; shift 2 ;;
    -y |--numprepproc   ) NUMPREPPROC="$2"                                                ; shift 2 ;;
    -g |--checkaccuracy ) GOLDEN=$VAI_ALVEO_ROOT/examples/deployment_modes/gold.txt       ; shift 1 ;;
    -v |--verbose       ) VERBOSE=1                                                       ; shift 1 ;;
    -x |--perpetual     ) PERPETUAL=1                                                     ; shift 1 ;;
    -cn|--customnet     ) CUSTOM_NETCFG="$2"                                              ; shift 2 ;;
    -cq|--customquant   ) CUSTOM_QUANTCFG="$2"                                            ; shift 2 ;;
    -cw|--customwts     ) CUSTOM_WEIGHTS="$2"                                             ; shift 2 ;;
    -p |--profile       ) PROFILING_ENABLE=1                                              ; shift 1 ;;
    -h |--help          ) usage                                                           ; exit  1 ;;
     *) break ;;
  esac
done

# Verbose Debug Profiling Prints
# Note, the VERBOSE mechanic here is not working
# Its always safer to set this manually
export XBLAS_EMIT_PROFILING_INFO=1
# To be fixed
export XBLAS_EMIT_PROFILING_INFO=$VERBOSE
export XDNN_VERBOSE=$VERBOSE

. ../../overlaybins/setup.sh

if [ "$MODEL" == "mobilenet" ]; then
  IMG_INPUT_SCALE=0.017
fi

# Determine XCLBIN
XCLBIN=/opt/xilinx/overlaybins/xdnnv3
if [ -d $XCLBIN ]; then
  echo "--- Using System XCLBIN ---"
else
  echo "--- Using Local XCLBIN ---"
  XCLBIN=${VAI_ALVEO_ROOT}/overlaybins/xdnnv3
fi
WEIGHTS=./data/${MODEL}_data.h5
DSP_WIDTH=96

NETCFG=./data/${MODEL}_8b_${COMPILEROPT}.json
QUANTCFG=./data/${MODEL}_8b_xdnnv3.json

if [ ! -z $CUSTOM_NETCFG ]; then
  NETCFG=$CUSTOM_NETCFG
fi
if [ ! -z $CUSTOM_WEIGHTS ]; then
  WEIGHTS=$CUSTOM_WEIGHTS
fi
if [ ! -z $CUSTOM_QUANTCFG ]; then
  QUANTCFG=$CUSTOM_QUANTCFG
fi

echo -e "Running:\n Test: $TEST\n Model: $MODEL\n Xclbin: $XCLBIN\n Accelerator: $ACCELERATOR\n"

BASEOPT="--xclbin $XCLBIN
         --netcfg $NETCFG
         --weights $WEIGHTS
         --labels $VAI_ALVEO_ROOT/examples/deployment_modes/synset_words.txt
         --quantizecfg $QUANTCFG
         --img_input_scale $IMG_INPUT_SCALE
         --batch_sz $BATCHSIZE"

if [ ! -z $GOLDEN ]; then
  BASEOPT+=" --golden $GOLDEN"
fi

##############################################
# auto-generate vitis rundir if not provided
##############################################
get_abs_filename() {
  # $1 : relative filename
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}
if [ -z $VITIS_RUNDIR ]; then
  find /tmp/vitis-{$USER} -mtime +3 -exec rm -fr {} \; &> /dev/null
  VITIS_RUNDIR=/tmp/vitis-${USER}/${BASHPID}
  mkdir -p $VITIS_RUNDIR
  rm -f ${VITIS_RUNDIR}/{compiler.json,quantizer.json,weights.h5}
  ln -s $(get_abs_filename $NETCFG) ${VITIS_RUNDIR}/compiler.json
  ln -s $(get_abs_filename $QUANTCFG) ${VITIS_RUNDIR}/quantizer.json
  ln -s $(get_abs_filename $WEIGHTS) ${VITIS_RUNDIR}/weights.h5
  echo "{ \"target\": \"xdnn\", \"filename\": \"\", \"kernel\": \"xdnn\", \"config_file\": \"\", \"lib\": \"${LIBXDNN_PATH}\", \"xclbin\": \"${XCLBIN}\", \"publish_id\": \"${BASHPID}\" }" > ${VITIS_RUNDIR}/meta.json
  # meta.json accepts {env_variables} in paths as well, e.g.:
  #echo "{ \"lib\": \"{VAI_ALVEO_ROOT}/vai/dpuv1/rt/xdnn_cpp/lib/libxfdnn.so\", \"xclbin\": \"{VAI_ALVEO_ROOT}/overlaybins/xdnnv3\" }" > ${VITIS_RUNDIR}/meta.json
  cp -fr $VITIS_RUNDIR ${VITIS_RUNDIR}_worker
  echo "{ \"target\": \"xdnn\", \"filename\": \"\", \"kernel\": \"xdnn\", \"config_file\": \"\", \"lib\": \"${LIBXDNN_PATH}\", \"xclbin\": \"${XCLBIN}\", \"subscribe_id\": \"${BASHPID}\" }" > ${VITIS_RUNDIR}_worker/meta.json
  BASEOPT+=" --vitis_rundir ${VITIS_RUNDIR}"
fi

# Build options for appropriate python script

####################
# single image test
####################

if [ "$TEST" == "test_classify" ]; then
  TEST=test_classify.py
  if [ -z ${DIRECTORY+x} ]; then
    DIRECTORY=dog.jpg
  fi
  BASEOPT+=" --images $DIRECTORY"

####################
# network profile
####################

elif [ "$TEST" == "profile" ]; then
  TEST=profile.py
  if [ -z ${DIRECTORY+x} ]; then
    DIRECTORY=dog.jpg
  fi
  BASEOPT+=" --images $DIRECTORY"

#######################################################
# multi-process streaming with xstream and vitis API
#######################################################
elif [[ "$TEST" == "streaming_classify_vitis"* ]] ; then
  if [ -z ${DIRECTORY+x} ]; then
    DIRECTORY=../../models/data/ilsvrc12/ilsvrc12_img_val
  fi

  if [ "$TEST" == "streaming_classify_vitis_fpgaonly" ]; then
    BASEOPT+=" --fpgaonly 1"
  elif [ "$TEST" == "streaming_classify_vitis_dataonly" ]; then
    BASEOPT+=" --dataonly 1"
  fi
  BASEOPT+=" --images $DIRECTORY"

  TEST=mp_classify_vitis.py

############################
# multi-process streaming
############################

elif [[ "$TEST" == "streaming_classify"*  || "$TEST" == "test_mp_classify"* ]] ; then
  if [ -z ${DIRECTORY+x} ]; then
    DIRECTORY=../../models/data/ilsvrc12/ilsvrc12_img_val
  fi

  if [ "$TEST" == "streaming_classify_fpgaonly" ]; then
    BASEOPT+=" --benchmarkmode 1"
  fi
  BASEOPT+=" --numstream $NUMSTREAMS"
  BASEOPT+=" --images $DIRECTORY"
  BASEOPT+=" --numprepproc $NUMPREPPROC"
  if [ "$PERPETUAL" == 1 ]; then
    BASEOPT+=" --zmqpub --perpetual --deviceID $DEVICEID"
  fi
  
  if [ "$PROFILING_ENABLE" == 1 ]; then
    BASEOPT+=" --profile"
  fi
  TEST=mp_classify.py

############################
# xstream service streaming
############################

elif [[ "$TEST" == "xstream"* ]] ; then
  if [ -z ${DIRECTORY+x} ]; then
    DIRECTORY=../../models/data/ilsvrc12/ilsvrc12_img_val
  fi

  if [[ "$TEST" == *"standalone"* ]]; then
    BASEOPT+=" --startxstream"
  fi

  if [[ "$TEST" == *"server"* ]]; then
    BASEOPT+=" --servermode"
  fi

  if [[ "$TEST" == *"fpgaonly"* ]]; then
    BASEOPT+=" --benchmarkmode 1"
  fi

  BASEOPT+=" --numstream $NUMSTREAMS"
  BASEOPT+=" --images $DIRECTORY"
  BASEOPT+=" --numprepproc $NUMPREPPROC"

  TEST=xs_classify.py
  rm -f xstream.log

################################################################
# multi-threaded streaming test for max multi-fpga performance
################################################################
elif [[ "$TEST" == "streaming_benchmark"* ]] ; then
  if [ -z ${DIRECTORY+x} ]; then
    DIRECTORY=../../models/data/ilsvrc12/ilsvrc12_img_val
  fi

  BASEOPT+=" --images $DIRECTORY"

  TEST=mt_benchmark.py

###########################
# switch to run the classification examples through c++ APIs
# runs with 8 bit quantization for now
###########################

elif [ "$TEST" == "classify_cpp" ]; then
  cd classify_cpp
  make
  cp ./classify.exe ../classify.exe
  cd -
  BATCHSIZE=1
  DIRECTORY=$VAI_ALVEO_ROOT/examples/deployment_modes/dog.jpg
  BASEOPT_CPP="--xclbin $XCLBIN --netcfg $NETCFG --datadir $WEIGHTS --labels ./synset_words.txt --quantizecfg $QUANTCFG --img_input_scale $IMG_INPUT_SCALE --batch_sz $BATCHSIZE"
  BASEOPT_CPP+=" --image $DIRECTORY"
  OPENCV_LIB=/usr/lib/x86_64-linux-gnu
  HDF5_PATH=${VAI_ALVEO_ROOT}/ext/hdf5
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VAI_ALVEO_ROOT/ext/zmq/libs:$VAI_ALVEO_ROOT/ext/boost/libs:${HDF5_PATH}/lib:$VAI_ALVEO_ROOT/vai/dpuv1/rt/libs:/opt/xilinx/xrt/lib:$OPENCV_LIB

###########################
# multi-PE multi-network (Run two different networks simultaneously)
# runs with 8 bit quantization for now
###########################

elif [ "$TEST" == "multinet" ]; then
  TEST=test_classify_async_multinet.py
  if [ -z ${DIRECTORY+x} ]; then
    DIRECTORY=dog.jpg
  fi
  BASEOPT+=" --images $DIRECTORY"
  BASEOPT+=" --dsp $DSP_WIDTH --jsoncfg data/multinet.json"
else
  echo "Test was improperly specified..."
  exit 1
fi

if [ "$TEST" == "classify_cpp" ]; then
  ./classify.exe $BASEOPT_CPP
else
  echo python $TEST $BASEOPT
  python $VAI_ALVEO_ROOT/examples/deployment_modes/$TEST $BASEOPT
fi
