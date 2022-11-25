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

# Supported Modes & Models


usage() {
  echo -e ""
  echo "Usage:"
  echo "------------------------------------------------"
  echo "  ./app_test.sh --xmodel_file  <xmodel-path>"
  echo "           --config_file  <path to config_file>"
  echo "           --image_dir    <image-dir>"
  echo "           --use_sw_pre_proc  (For software Preprocessing)"
  echo "           --use_sw_post_proc (For Software Postprocessing)"
  echo "           --performance_diff    (To compare the Performance of Software and Hardware pre and post processing)"
  echo "           --verbose      (To print detection outputs contains the lable, coordinates and confidence values for every input image) "
  echo -e ""

  }

# Defaults
config_file=""
xmodel_file=""
img_dir=""
sw_pre_proc=0
sw_post_proc=0
verbose=0
performance_diff=0
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

  if [[ "$1" != "--use_sw_pre_proc" && "$1" != "--verbose" && "$1" != "--performance_diff" && "$1" != "--use_sw_post_proc" && -z "$2" ]]; then
    echo -e "\n[ERROR] Missing argument value for $1 \n";
    exit 1;
  fi
  case "$1" in
    --config_file        ) config_file="$2"                ; shift 2 ;; 
    --xmodel_file        ) xmodel_file="$2"                ; shift 2 ;;
    --image_dir          ) img_dir="$2"                    ; shift 2 ;;
    --use_sw_pre_proc    ) sw_pre_proc=1                   ; shift 1 ;;
    --use_sw_post_proc   ) sw_post_proc=1                  ; shift 1 ;;
    --verbose            ) verbose=1                       ; shift 1 ;;
    --performance_diff   ) performance_diff=1              ; shift 1 ;;   
     *) echo "Unknown argument : $1";
        echo "Try ./app_test.sh -h to get correct usage. Exiting ...";
        exit 1 ;;
  esac
done

if [[ "$xmodel_file" =  "" ]]; then
  echo -e ""
  echo -e "[ERROR] No xmodel file selected !"
  echo -e "[ERROR] Check Usage with: ./app_test.sh -h "
  echo -e ""
  exit 1
fi

if [[ "$img_dir" =  "" ]]; then
  echo -e ""
  echo -e "[ERROR] No image directory selected !"
  echo -e "[ERROR] Check Usage with: ./app_test.sh -h "
  echo -e ""
  exit 1
fi

if [[ "$config_file" =  "" ]]; then
  echo -e ""
  echo -e "[ERROR] No config file provided !"
  echo -e "[ERROR] Check Usage with: ./app_test.sh -h "
  echo -e ""
  exit 1
fi


CPP_EXE="./app.exe"


if [ "$performance_diff" -eq 0 ]; 
then
 exec_args="$config_file $xmodel_file $img_dir $sw_pre_proc $sw_post_proc $verbose"
 ${CPP_EXE} ${exec_args} 
else
 echo -e "\n Running Application with Software Preprocessing and Postprocessing \n"
 sw_pre_proc=1
 sw_post_proc=1
 verbose=0
 
 exec_args="$config_file $xmodel_file $img_dir $sw_pre_proc $sw_post_proc $verbose"
 ${CPP_EXE} ${exec_args} |& grep -e "E2E Performance" -e "Pre-process Latency" -e "Execution Latency" -e "Post-process Latency" > z.log
 grep "E2E Performance" z.log > x.log
 grep "Pre-process Latency" z.log > x1.log 
 grep "Execution Latency" z.log > x2.log 
 grep "Post-process Latency" z.log > x3.log
 
 awk '{print $3 > "xx.log"}' x.log
 awk '{print $3 > "xx1.log"}' x1.log  
 awk '{print $3 > "xx2.log"}' x2.log 
 awk '{print $3 > "xx3.log"}' x3.log 

 read i<xx.log
 read a<xx1.log
 read b<xx2.log
 read c<xx3.log

 i=$(printf "%.2f" $i)
 a=$(printf "%.2f" $a)
 b=$(printf "%.2f" $b)
 c=$(printf "%.2f" $c)

 printf "   E2E Performance: %.2f fps\n" $i
 printf "   Pre-process Latency: %.2f ms\n" $a
 printf "   Execution Latency: %.2f ms\n" $b
 printf "   Post-process Latency: %.2f ms" $c

 echo -e "\n"
 
 rm z.log
 rm x.log
 rm xx.log
 rm x1.log
 rm xx1.log
 rm x2.log
 rm xx2.log
 rm x3.log
 rm xx3.log

 echo -e " Running Application with Hardware Preprocessing and Postprocessing\n"
 sw_pre_proc=0
 sw_post_proc=0
 exec_args="$config_file $xmodel_file $img_dir $sw_pre_proc $sw_post_proc $verbose"
 ${CPP_EXE} ${exec_args} |& grep -e "E2E Performance" -e "Pre-process Latency" -e "Execution Latency" -e "Post-process Latency" > z1.log
 
 grep "E2E Performance" z1.log > y.log
 grep "Pre-process Latency" z1.log > y1.log 
 grep "Execution Latency" z1.log > y2.log 
 grep "Post-process Latency" z1.log > y3.log  

 awk '{print $3 > "yy.log"}' y.log
 awk '{print $3 > "yy1.log"}' y1.log 
 awk '{print $3 > "yy2.log"}' y2.log 
 awk '{print $3 > "yy3.log"}' y3.log 

 read j<yy.log
 read a<yy1.log
 read b<yy2.log
 read c<yy3.log

 j=$(printf "%.2f" $j)
 a=$(printf "%.2f" $a)
 b=$(printf "%.2f" $b)
 c=$(printf "%.2f" $c)

 k=`expr "$j - $i" |bc` 
 f=`expr $k*100 |bc`
 printf "   E2E Performance: %.2f fps\n" $j
 printf "   Pre-process Latency: %.2f ms\n" $a
 printf "   Execution Latency: %.2f ms\n" $b
 printf "   Post-process Latency: %.2f ms" $c

 h=`expr $f/$i |bc -l`
 echo -e "\n"
 printf "   The percentage improvement in throughput is %.2f" $h 
 echo -e " %\n"
 rm z1.log
 rm y.log
 rm yy.log
 rm y1.log
 rm yy1.log
 rm y2.log
 rm yy2.log
 rm y3.log
 rm yy3.log
fi









