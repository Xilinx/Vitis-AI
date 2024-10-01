#!/bin/bash
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

set -e

script_path=$(dirname "$(realpath $0)")
project_name=$(basename ${script_path})

# cmake args
declare -a args
args=(-DBUILD_TEST=ON)
args+=(-DBUILD_SHARED_LIBS=ON)

# parse options
options=$(getopt -a -n 'parse-options' -o h \
		 -l help,clean,build-only,type:,pack:,build-dir:,install-prefix:,cmake-options: \
		 -- "$0" "$@")
[ $? -eq 0 ] || {
    echo "Failed to parse arguments! try --help"
    exit 1
}
eval set -- "$options"
while true; do
    case "$1" in
	-h | --help) show_help=true; break;;
	--clean) clean=true;;
	--build-only) build_only=true;;
	--type)
	    shift
	    case "$1" in
		release) build_type=Release;;
		debug) build_type=Debug;;
		*) echo "Invalid build type \"$1\"! try --help"; exit 1;;
	    esac
	    ;;
	--pack)
	    shift
            build_package=true
	    case "$1" in
		deb) args+=(-DCPACK_GENERATOR="DEB");;
		rpm) args+=(-DCPACK_GENERATOR="RPM");;
		*) echo "Invalid pack format \"$1\"! try --help"; exit 1;;
	    esac
	    ;;
	--build-dir) shift; build_dir=$1;;
	--install-prefix) shift; install_prefix=$1;;
	--cmake-options) shift; args+=($1);;
	--) shift; break;;
    esac
    shift
done

# set build type
args+=(-DCMAKE_BUILD_TYPE=${build_type:="Debug"})
[ ${build_type} == "Debug" ] && args+=(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)

# detect target & set install prefix
if [ -z ${OECORE_TARGET_SYSROOT:+x} ]; then
    echo "Native-platform building..."
    os=`lsb_release -a | grep "Distributor ID" | sed 's/^.*:\s*//'`
    os_version=`lsb_release -a | grep "Release" | sed 's/^.*:\s*//'`
    arch=`uname -p`
    target_info=${os}.${os_version}.${arch}.${build_type}
    install_prefix_default=$HOME/.local/${target_info}
else
    echo "Cross-platform building..."
    echo "Found target sysroot ${OECORE_TARGET_SYSROOT}"
    target_info=${OECORE_TARGET_OS}.${OECORE_SDK_VERSION}.${OECORE_TARGET_ARCH}.${build_type}
    install_prefix_default=${OECORE_TARGET_SYSROOT}/install/${build_type}
    args+=(-DCMAKE_TOOLCHAIN_FILE=${OECORE_NATIVE_SYSROOT}/usr/share/cmake/OEToolchainConfig.cmake)
    args+=(-DCPACK_DEBIAN_PACKAGE_ARCHITECTURE=arm64)
fi
args+=(-DCMAKE_INSTALL_PREFIX=${install_prefix:="${install_prefix_default}"})

# set build dir
build_dir_default=$HOME/build/build.${target_info}/${project_name}
[ -z ${build_dir:+x} ] && build_dir=${build_dir_default}

if [ ${show_help:=false} == true ]; then
    echo "./cmake.sh [options]"
    echo "    --help                    show help"
    echo "    --clean                   discard build dir before build"
    echo "    --build-only              build only, will not install"
    echo "    --type[=TYPE]             build type. VAR {release, debug(default)}"
    echo "    --pack[=FORMAT]           enable packing and set package format. VAR {deb, rpm}"
    echo "    --build-dir[=DIR]         set customized build directory. default directory is ${build_dir_default}"
    echo "    --install-prefix[=PREFIX] set customized install prefix. default prefix is ${install_prefix_default}"
    echo "    --cmake-options[=OPTIONS] append more cmake options"
else
    mkdir -p ${build_dir}
    ${clean:=false} && rm -fr ${build_dir}/*
    cd -P ${build_dir}
    echo "cd $PWD"
    echo cmake "${args[@]}" "$script_path"
    cmake "${args[@]}" "$script_path"
    make -j
    ${build_only:=false} || make install
    ${build_package:=false} && make package
fi

exit 0
