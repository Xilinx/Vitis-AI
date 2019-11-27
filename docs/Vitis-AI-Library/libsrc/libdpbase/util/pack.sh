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

