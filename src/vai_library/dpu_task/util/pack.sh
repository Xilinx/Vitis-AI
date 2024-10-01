#
# Copyright 2022-2023 Advanced Micro Devices Inc.
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
#!/bin/bash
if [ $# == 0 ]
then
  echo "Usage: $0 build_dir" 
  exit 
fi

work_path=`pwd`
build_dir=$1
build_dir=${build_dir%%/}
target="libdpbase"

pack_dir=${target}_${build_dir}
if [ -d ${pack_dir} ]
then
  rm ${pack_dir} -rf
fi
mkdir -p ${pack_dir}
cp ${build_dir}/install/lib ${pack_dir}/lib -rf
#cp ${build_dir}/install/include ${pack_dir}/include -rf
mkdir -p ${pack_dir}/test
mkdir -p ${pack_dir}/bin

test_cases=`ls ${build_dir} -1|grep test`
for c in ${test_cases}
do
  if [ -x ${build_dir}/${c} -a ! -d ${build_dir}/${c} ] 
  then
    cp ${build_dir}/${c} ${pack_dir}/bin/
  fi
done

cp util/run.sh ${pack_dir}

