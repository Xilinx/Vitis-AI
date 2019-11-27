#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
LD_LIBRARY_PATH=${DIR}/lib/:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH
cd ./bin
test_cases=`ls -1|grep test`
for c in ${test_cases}
do
  if [ -x ${c} ]
  then
     ./$c
  fi
done


