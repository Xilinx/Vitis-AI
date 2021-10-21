#!/bin/bash

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


TOTALTESTS=0
TOTALFAILS=0
NUMFILES=0

for f in res*.txt
do
    NUMFILES=$(($NUMFILES + 1))
    NUMTESTS=0
    NUMFAILS=0
    while IFS= read -r line
    do
        case "$line" in
            *Tests*) X=$(echo $line | cut -d'=' -f2)
                NUMTESTS=$(($NUMTESTS + $X))
                TOTALTESTS=$(($TOTALTESTS + $X))
             ;;
            *Fails*) X=$(echo $line | cut -d'=' -f2)
                NUMFAILS=$(($NUMFAILS + $X))
                TOTALFAILS=$(($TOTALFAILS + $X))
             ;;
        esac
    done < "$f"
    echo $f
    echo Num Fails = $NUMFAILS
    echo

done
    
echo
echo SUMMARY
echo Num files          = $NUMFILES
echo Num tests per file = $NUMTESTS
echo
echo Total Tests = $TOTALTESTS
echo Total Fails = $TOTALFAILS
