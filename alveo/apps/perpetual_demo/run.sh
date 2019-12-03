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
#python server.py -z 127.0.0.1:5505,5515,5525,5535,5545,5555,5565,5575 -x 127.0.0.1:5506,5516,5526,5536,5546,5556,5566,5576

# User must set this - maybe in future we can auto-detect
# aws : 1
# alveo-u200 : 2
# alveo-u200-ml : 3
# alveo-u250 : 4
# 1525 : 2
NUMPE=2

DIRECTORY=$HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min

ln -sf $DIRECTORY www/imagenet_val

NUMCARDS=`/opt/xilinx/xrt/bin/xbutil list | wc -l`
let "NUMCARDS = $NUMCARDS -1"

sed -i "s/var numDevices.*$/var numDevices = ${NUMCARDS};/g" www/index.html
sed -i "s/num_cores:.*$/num_cores: ${NUMPE},/g" www/index.html

echo ""
echo "Go to a web browser and paste:"
echo "http://${HOSTNAME}:8998/static/www/index.html#$NUMCARDS"
echo ""
echo "To kill the demo, use CTRL+C, and then:"
echo 'kill -9 $(ps -s $$ -o pid=)'
echo ""

let "NUMCARDS = $NUMCARDS -1"
PORTSZ=""
PORTSX=""
cd $VAI_ALVEO_ROOT/examples/deployment_modes
for i in $(seq 0 $NUMCARDS)
  do
    PORTSZ="${PORTSZ}55${i}5"
    PORTSX="${PORTSX}55${i}6"
    if [[ $i != $NUMCARDS ]]; then
      PORTSZ="${PORTSZ},"
      PORTSX="${PORTSX},"
    fi
    ./run.sh -t streaming_classify -i $i -x -v -c throughput -ns 8 -d $DIRECTORY > /dev/null &
  done

cd -
python server.py -z 127.0.0.1:$PORTSZ -x 127.0.0.1:$PORTSX > /dev/null
