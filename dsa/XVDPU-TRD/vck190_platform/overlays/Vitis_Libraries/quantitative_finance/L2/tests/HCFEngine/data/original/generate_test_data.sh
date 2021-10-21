#!/bin/bash
#
# Copyright 2019 Xilinx, Inc.
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
#

for f in tc_*.txt
do
    filename=$(basename -- "$f")
    filename="${filename%.*}"

    TC=$f
    QL=$(echo $filename | sed "s/tc/ql/").csv
    OP=$(echo $filename | sed "s/tc/test_data/").txt

    # Use this line for full grid
    ./generate_test_data $TC $QL $OP

    # Use this line to create a sub grid -s<min_s0 (default 0)> -S<max_s0 (default 800)> -v<min_v0 (default 0)> -V<max_v0 (default 5)>
    #./generate_test_data -s20 -S200 -v0.1 -V2 $TC $QL $OP
done

