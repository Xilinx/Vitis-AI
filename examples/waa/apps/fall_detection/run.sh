#!/usr/bin/env bash

usage() {
  echo -e ""
  echo "Usage:"
  echo "------------------------------------------------"
  echo "    ./run.sh --dir <path-to-directory> --num-threads <num-of-threads> --verbose <AKS_VERBOSE>"
  echo -e ""
  echo "Arguments:"
  echo "-----------------------------------------------------------------------------"
  echo "  -d  DIRECTORY                | --dir          DIRECTORY                Images/videos directory"
  echo "  -a  AKS_ROOT                 | --aks-root     AKS_ROOT                 Path to AKS root directory"
  echo "  -t  MAX_NUM_CONCURRENT_STREAMS   | --num-threads  MAX_NUM_CONCURRENT_STREAMS   Path to AKS root directory"
  echo "  -v  VERBOSE                  | --verbose      VERBOSE                  Verbosity level"
  echo "                                   Possible values: [0 - Only Warnings & Errors]"
  echo "                                   Possible values: [1 - Important Information, warnings & errors]"
  echo "                                   Possible values: [2 - All debug, performance metrics, warnings & errors]"
  echo "  -h                           | --help                                  Print this message."
  echo "  -run_sw                      | --run_sw_optical_flow                   To run optical flow with software."
  echo "  -perf_diff                   | --performance_diff                      To compare the performance with software and hardware optical flow"
  echo "  -accu_diff                   | --accuracy_diff                         To compare the accuracy with software and hardware optical flow"
  echo "-----------------------------------------------------------------------------"
  echo -e ""
}

# Default
DIRECTORY=""
MAX_NUM_CONCURRENT_STREAMS=70
VERBOSE=1
SW_OPTICAL_FLOW=0
PERFORMANCE_DIFF=0
ACCURACY_DIFF=0
AKS_ROOT=${VAI_HOME}/src/AKS

# Parse Options
while true
do
  if [[ -z "$1" ]]; then break; fi
  case "$1" in
    -d          |--dir                  ) DIRECTORY="$2"                       ; shift 2 ;;
    -t          |--num-threads          ) MAX_NUM_CONCURRENT_STREAMS="$2"      ; shift 2 ;;
    -a          |--aks-root             ) AKS_ROOT="$2"                        ; shift 2 ;;
    -run_sw     |--run_sw_optical_flow  ) SW_OPTICAL_FLOW=1                    ; shift 1 ;;
    -perf_diff  |--performance_diff     ) PERFORMANCE_DIFF=1                   ; shift 1 ;;
    -accu_diff  |--accuracy_diff        ) ACCURACY_DIFF=1                      ; shift 1 ;;
    -v          |--verbose              ) VERBOSE="$2"                         ; shift 2 ;;
    -h          |--help                 ) usage                                ; exit  1 ;;
     *) echo "Unknown argument : $1";
        echo "Try ./run.sh -h to get correct usage. Exiting ...";
        exit 1 ;;
  esac
done

if [ ! -d "${AKS_ROOT}" ]
then
  echo -e "Path '${AKS_ROOT}' doesn't exist. Please specify a valid path to AKS."
  exit 1
fi

export AKS_ROOT
# Add verbose level
export AKS_VERBOSE=$VERBOSE

export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/waa_u200_xclbins_v2_0_0/fall_detection/dpu.xclbin

if [ -z ${DIRECTORY} ];
then
  usage;
  exit 1;
fi

AKS_GRAPH_META_URL="https://www.xilinx.com/bin/public/openDownload?filename=fall_detection.tar.gz"
NAME="model/fall_detection.xmodel"
if [[ ! -f "${NAME}" ]]; then
    echo -e "$NAME doesn't exist"
    wget $AKS_GRAPH_META_URL -O fall_detection.tar.gz && tar -xzvf fall_detection.tar.gz && rm fall_detection.tar.gz
    if [[ $? != 0 ]]; then echo "Network download failed. Exiting ..."; exit 1; fi;
fi;

# Add Library Paths
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:libs
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${AKS_ROOT}

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}

 
if [[ "$PERFORMANCE_DIFF" -eq 0 && "$ACCURACY_DIFF" -eq 0 ]]; 
then
 ./src/bin/main.exe $DIRECTORY $MAX_NUM_CONCURRENT_STREAMS $SW_OPTICAL_FLOW 
fi

if [ "$PERFORMANCE_DIFF" -eq 1 ];
then 
 echo -e "\n Running Performance Diff: "
 echo -e "\n   Running Application with Software Optical Flow \n"
 SW_OPTICAL_FLOW=1
 ./src/bin/main.exe $DIRECTORY $MAX_NUM_CONCURRENT_STREAMS $SW_OPTICAL_FLOW |& grep "Throughput (fps)" > x.log
 awk '{print $3 > "xx.log"}' x.log 
 read i<xx.log
 i=$(printf "%.2f" $i)
 printf "   Throughput : %.2f fps" $i
 echo -e "\n"
 rm x.log
 rm xx.log
 echo -e "   Running Application with Hardware Optical Flow \n"
 SW_OPTICAL_FLOW=0
 ./src/bin/main.exe $DIRECTORY $MAX_NUM_CONCURRENT_STREAMS $SW_OPTICAL_FLOW |& grep "Throughput (fps)" > y.log
 awk '{print $3 > "yy.log"}' y.log
 read j<yy.log
 j=$(printf "%.2f" $j)
 k=`expr "$j - $i" |bc` 
 f=`expr $k*100 |bc`
 printf "   Throughput : %.2f fps" $j
 h=`expr $f/$i |bc -l`
 echo -e "\n"
 printf "   The percentage improvement in throughput is %.2f" $h 
 echo -e " %\n"
 rm y.log
 rm yy.log
fi

if [ "$ACCURACY_DIFF" -eq 1 ]; 
then
 echo -e "\n Running Accuracy Diff: "
 echo -e "\n   Running Application with Software Optical Flow \n"
 SW_OPTICAL_FLOW=1
 ./src/bin/main.exe $DIRECTORY $MAX_NUM_CONCURRENT_STREAMS $SW_OPTICAL_FLOW |& grep "Accuracy:" > x.log
 awk '{print $2 > "xx.log"}' x.log 
 read i<xx.log
 i=`expr $i*100 |bc`
 i=$(printf "%.2f" $i)
 printf "   Accuracy of the network is %.2f %%" $i
 echo -e "\n"
 rm x.log
 rm xx.log
 echo -e "   Running Application with Hardware Optical Flow \n"
 SW_OPTICAL_FLOW=0
 ./src/bin/main.exe $DIRECTORY $MAX_NUM_CONCURRENT_STREAMS $SW_OPTICAL_FLOW |& grep "Accuracy:" > y.log
 awk '{print $2 > "yy.log"}' y.log
 read j<yy.log
 j=`expr $j*100 |bc`
 j=$(printf "%.2f" $j)
 k=`expr "$j - $i" |bc` 
 f=`expr $k*100 |bc`
 printf "   Accuracy of the network is %.2f %%" $j
 h=`expr $f/$i |bc -l`
 echo -e "\n"
 printf "   The percentage improvement in accuracy is %.2f " $h 
 echo -e " %\n"
 rm y.log
 rm yy.log
fi
