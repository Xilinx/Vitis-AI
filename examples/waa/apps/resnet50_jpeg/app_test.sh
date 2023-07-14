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
  echo "           --image_dir    <image-dir>"
  echo "           --use_sw_pre_proc  (For software Preprocessing)"
  echo "           --no_zero_copy (To disable zero copy)"
  echo "           --label_file   <label-file>"
  echo "           --verbose      (To print Top 5 outputs for each image) "
  echo "           --performance_diff    (To compare the Performance of Software and Hardware preprocessing)"
  echo "           --accuracy_diff    (To compare the Accuracy of Software and Hardware preprocessing)"
  echo -e ""

  }

# Defaults
xmodel_file=""
img_dir=""
sw_proc=0
no_zcpy=0
label_file=""
verbose=0
performance_diff=0
accuracy_diff=0

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

  if [[ "$1" != "--use_sw_pre_proc" && "$1" != "--verbose" && "$1" != "--no_zero_copy" && "$1" != "--performance_diff" && "$1" != "--accuracy_diff" && -z "$2" ]]; then
    echo -e "\n[ERROR] Missing argument value for $1 \n";
    exit 1;
  fi
  case "$1" in
    --xmodel_file        ) xmodel_file="$2"                ; shift 2 ;;
    --image_dir          ) img_dir="$2"                    ; shift 2 ;;
    --use_sw_pre_proc    ) sw_proc=1                       ; shift 1 ;;
    --no_zero_copy       ) no_zcpy=1                       ; shift 1 ;;
    --label_file         ) label_file="$2"                 ; shift 2 ;;
    --verbose            ) verbose=1                       ; shift 1 ;;
    --performance_diff   ) performance_diff=1              ; shift 1 ;;
    --accuracy_diff      ) accuracy_diff=1                 ; shift 1 ;;
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


CPP_EXE="./bin/resnet50_jpeg.exe"

if [[ "$performance_diff" -eq 0 && "$accuracy_diff" -eq 0 ]]; 
then
exec_args="$xmodel_file $img_dir $sw_proc $no_zcpy $verbose $label_file"
${CPP_EXE} ${exec_args} 
fi

if [ "$performance_diff" -eq 1 ];
then 
 echo -e "\n Running Performance Diff: "
 echo -e "\n   Running Application with Software Preprocessing \n"
 sw_proc=1
 no_zcpy=1
 verbose=0
 exec_args="$xmodel_file $img_dir $sw_proc $no_zcpy $verbose $label_file"
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

 echo -e "   Running Application with Hardware Preprocessing \n"
 sw_proc=0
 no_zcpy=0
 verbose=0
 exec_args="$xmodel_file $img_dir $sw_proc $no_zcpy $verbose $label_file"
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

 k=$(awk -vn1="$j" -vn2="$i" 'BEGIN{ print ( n1 - n2) }')
 f=$(awk -vn1="$k" -vn2="100" 'BEGIN{ print ( n1 * n2) }')

 printf "   E2E Performance: %.2f fps\n" $j
 printf "   Pre-process Latency: %.2f ms\n" $a
 printf "   Execution Latency: %.2f ms\n" $b
 printf "   Post-process Latency: %.2f ms" $c

 h=$(awk -vn1="$f" -vn2="$i" 'BEGIN{ print ( n1 / n2) }')
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

if [ "$accuracy_diff" -eq 1 ]; 
then
if [[ "$label_file" = "" ]];
then
echo -e ""
echo -e "[ERROR] No label file selected !"
echo -e ""
exit 1
fi
 echo -e "\n Running Accuracy Diff: "
 echo -e "\n   Running Application with Software Preprocessing \n"
 sw_proc=1
 no_zcpy=1
 verbose=0
 exec_args="$xmodel_file $img_dir $sw_proc $no_zcpy $verbose $label_file"
 ${CPP_EXE} ${exec_args} |& grep "accuracy of the network" > x.log
 awk '{print $7 > "xx.log"}' x.log 
 read i<xx.log
 i=$(printf "%.2f" $i)
 printf "   Accuracy of the network is %.2f %%" $i
 echo -e "\n"
 rm x.log
 rm xx.log
 echo -e "   Running Application with Hardware Preprocessing \n"
 sw_proc=0
 no_zcpy=0
 verbose=0
 exec_args="$xmodel_file $img_dir $sw_proc $no_zcpy $verbose $label_file"
 ${CPP_EXE} ${exec_args} |& grep "accuracy of the network" > y.log
 awk '{print $7 > "yy.log"}' y.log
 read j<yy.log
 j=$(printf "%.2f" $j)
 k=$(awk -vn1="$j" -vn2="$i" 'BEGIN{ print ( n1 - n2) }')
 f=$(awk -vn1="$k" -vn2="100" 'BEGIN{ print ( n1 * n2) }')
 printf "   Accuracy of the network is %.2f %%" $j
 h=$(awk -vn1="$f" -vn2="$i" 'BEGIN{ print ( n1 / n2) }')
 echo -e "\n"
 printf "   The percentage improvement in accuracy is %.2f " $h 
 echo -e " %\n"
 rm y.log
 rm yy.log
fi
