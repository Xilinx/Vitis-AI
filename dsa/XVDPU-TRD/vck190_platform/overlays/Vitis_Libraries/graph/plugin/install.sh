#!/bin/bash
#
# Copyright 2020-2021 Xilinx, Inc.
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

set -e
echo "Checking TigerGraph installation version and directory"
if ! [ -x "$(command -v jq)" ]; then
    echo "ERROR: The program jq is required. Please follow the instructions below to install it:"
    echo "       RedHat/CentOS: sudo yum install jq"
    echo "       Ubuntu: sudo apt-get install jq"
    exit 1
fi

if ! [ -x "$(command -v gadmin)" ]; then
    echo "ERROR: Cannot find TigerGraph installation. Please run this install script as the user for the TigerGraph installation."
    exit 1
fi

if [ ! -f "$HOME/.tg.cfg" ]; then
    echo "ERROR: This script only supports TigerGraph version 3.x"
    echo "INFO: Installed version:"
    gadmin version | grep TigerGraph
    exit 1
fi
tg_root_dir=$(cat $HOME/.tg.cfg | jq .System.AppRoot | tr -d \")
tg_temp_root=$(cat $HOME/.tg.cfg | jq .System.TempRoot | tr -d \")
echo "INFO: Found TigerGraph installation in $tg_root_dir"
echo "INFO: TigerGraph TEMP root is $tg_temp_root"

rm -rf tigergraph/QueryUdf/tgFunctions.hpp
rm -rf tigergraph/QueryUdf/ExprFunctions.hpp
rm -rf tigergraph/QueryUdf/ExprUtil.hpp
rm -rf tigergraph/QueryUdf/graph.hpp

# save a copy of the original UDF Files
if [ ! -d "$tg_root_dir/dev/gdk/gsql/src/QueryUdf.orig" ]; then
    cp -r $tg_root_dir/dev/gdk/gsql/src/QueryUdf $tg_root_dir/dev/gdk/gsql/src/QueryUdf.orig
    echo "Saved a copy of the original QueryUdf files in $tg_root_dir/gdk/gsql/src/QueryUdf.orig"
fi

# prepare UDF files
cp $tg_root_dir/dev/gdk/gsql/src/QueryUdf.orig/ExprFunctions.hpp tigergraph/QueryUdf/tgFunctions.hpp
cp $tg_root_dir/dev/gdk/gsql/src/QueryUdf.orig/ExprUtil.hpp tigergraph/QueryUdf/
cp tigergraph/QueryUdf/xilinxUdf.hpp tigergraph/QueryUdf/ExprFunctions.hpp
cp -rf ../L3/include/graph.hpp tigergraph/QueryUdf/

xrtPath=/opt/xilinx/xrt
xrmPath=/opt/xilinx/xrm

while getopts ":r:m:" opt
do
case $opt in
    r)
    xrtPath=$OPTARG
    echo "$xrtPath"
    ;;
    m)
    xrmPath=$OPTARG
    echo "$xrmPath"
    ;;
    ?)
    echo "unknown"
    exit 1;;
    esac
done

source $xrtPath/setup.sh
source $xrmPath/setup.sh

# make L3 wrapper library
#make clean
make TigerGraphPath=$tg_root_dir libgraphL3wrapper

# copy files to $tg_rrot_dir UDF area
mkdir -p $tg_temp_root/gsql/codegen/udf
timestamp=$(date "+%Y%m%d-%H%M%S")
#rm -rf $tg_install_dir/tigergraph/dev/gdk/gsql/src/QueryUdf
cp -rf tigergraph/QueryUdf/ExprFunctions.hpp $tg_root_dir/dev/gdk/gsql/src/QueryUdf/
cp -rf tigergraph/QueryUdf/ExprUtil.hpp $tg_root_dir/dev/gdk/gsql/src/QueryUdf/

cp -rf tigergraph/QueryUdf/codevector.hpp $tg_root_dir/dev/gdk/gsql/src/QueryUdf/
cp -rf tigergraph/QueryUdf/loader.hpp $tg_root_dir/dev/gdk/gsql/src/QueryUdf/
cp -rf tigergraph/QueryUdf/tgFunctions.hpp $tg_root_dir/dev/gdk/gsql/src/QueryUdf/
cp -rf tigergraph/QueryUdf/graph.hpp $tg_root_dir/dev/gdk/gsql/src/QueryUdf/
cp -rf tigergraph/QueryUdf/core.cpp $tg_root_dir/dev/gdk/gsql/src/QueryUdf/

cp -rf tigergraph/QueryUdf/tgFunctions.hpp $tg_temp_root/gsql/codegen/udf
cp -rf tigergraph/QueryUdf/loader.hpp $tg_temp_root/gsql/codegen/udf
cp -rf tigergraph/QueryUdf/graph.hpp $tg_temp_root/gsql/codegen/udf
cp -rf tigergraph/QueryUdf/codevector.hpp $tg_temp_root/gsql/codegen/udf

cp -rf tigergraph/QueryUdf/*.json $tg_root_dir/dev/gdk/gsql/src/QueryUdf/
cp -rf tigergraph/QueryUdf/libgraphL3wrapper.so $tg_root_dir/dev/gdk/gsql/src/QueryUdf/
cp $tg_root_dir/dev/gdk/MakeUdf $tg_root_dir/dev/gdk/MakeUdf-$timestamp
cp -rf tigergraph/MakeUdf $tg_root_dir/dev/gdk/

# update files with tg_root_dir
for f in $tg_root_dir/dev/gdk/gsql/src/QueryUdf/*.json; do
    # use | as the demiliter since tg_root_dir has / in it
    sed -i "s|TG_ROOT_DIR|$tg_root_dir|" $f 
done
sed -i "s|TG_ROOT_DIR|$tg_root_dir|" $tg_root_dir/dev/gdk/MakeUdf 

gadmin start all
gadmin config set GPE.BasicConfig.Env "LD_PRELOAD=\$LD_PRELOAD; LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:/opt/xilinx/xrm/lib:/usr/lib/x86_64-linux-gnu/:\$LD_LIBRARY_PATH; CPUPROFILE=/tmp/tg_cpu_profiler; CPUPROFILESIGNAL=12; MALLOC_CONF=prof:true,prof_active:false; XILINX_XRT=/opt/xilinx/xrt; XILINX_XRM=/opt/xilinx/xrm"

echo "Apply the new configurations"
gadmin config apply -y
gadmin restart gpe -y
gadmin config get GPE.BasicConfig.Env

echo "Xilinx FPGA acceleration plugin for Tigergraph has been installed."

# Copy xclbins to TG root directory 
echo "Xilinx FPGA binary files for accelerated graph functions need to be dowloaded"
echo "from Xilinx DBA lounge and then installed by following instructions in the package."
#mkdir -p $tg_root_dir/dev/gdk/gsql/src/QueryUdf/xclbin
#
