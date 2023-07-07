#!/bin/bash
#
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
#

set -e

script_path=$(dirname "$(realpath $0)")
project_name=$(basename ${script_path})

function    usage() {
    echo "./cmake.sh [options]"
    echo "    --help                    show help"
    echo "    --clean                   discard build dir before build"
    echo "    --build-only              build only, will not install"
    echo "    --build-python            build python. if --pack is declared, will build conda package"
    echo "    --type[=TYPE]             build type. VAR {release, debug(default)}"
    echo "    --pack[=FORMAT]           enable packing and set package format. VAR {deb, rpm}"
    echo "    --build-dir[=DIR]         set customized build directory. default directory is ${build_dir_default}"
    echo "    --src-dir[=DIR]           set customized src directory. default directory is ${src_dir_default}"
    echo "    --install-prefix[=PREFIX] set customized install prefix. default prefix is ${install_prefix_default}"
    echo "    --cmake-options[=OPTIONS] append more cmake options"
    exit 0
}

# cmake args
declare -a args
args=(-DBUILD_TEST=ON)
args+=(-DBUILD_SHARED_LIBS=ON)
args+=(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
args+=(-DCMAKE_BUILD_TYPE=Debug)
build_type=Debug
# parse options
options=$(getopt -a -n 'parse-options' -o h \
		         -l help,ninja,build-python,clean,build-only,type:,project:,pack:,build-dir:,src-dir:,install-prefix:,cmake-options:,home,user \
		         -- "$0" "$@")
[ $? -eq 0 ] || {
    echo "Failed to parse arguments! try --help"
    exit 1
}
eval set -- "$options"
while true; do
    case "$1" in
        -h | --help) show_help=true; usage; break;;
	    --clean) clean=true;;
	    --build-only) build_only=true;;
	    --type)
	        shift
	        case "$1" in
                release)
                    build_type=Release;
                    args+=(-DCMAKE_BUILD_TYPE=${build_type:="Release"});;
                debug)
                    build_type=Debug;
                    args+=(-DCMAKE_BUILD_TYPE=${build_type:="Debug"});;
		        *) echo "Invalid build type \"$1\"! try --help"; exit 1;;
	        esac
	        ;;
	    --pack)
	        shift
                build_package=true
                cpack_generator=
	        case "$1" in
		        deb)
                            cpack_generator=DEB;
                            args+=(-DCPACK_GENERATOR=${cpack_generator});;
		        rpm)
                            cpack_generator=RPM;
                            args+=(-DCPACK_GENERATOR=${cpack_generator});;
		        *) echo "Invalid pack format \"$1\"! try --help"; exit 1;;
	        esac
	        ;;
	    --build-dir) shift; build_dir="$(realpath $1)";;
	    --src-dir) shift; src_dir=$1;;
	    --install-prefix) shift; install_prefix=$1;;
	    --cmake-options) shift; args+=($1);;
        --ninja) args+=(-G Ninja);;
        --project) shift;script_path="$(realpath ../$1)"; project_name=$1;;
        --build-python) args+=(-DBUILD_PYTHON=ON);;
	    --user) args+=(-DINSTALL_USER=ON);;
	    --home) args+=(-DINSTALL_HOME=ON);;
	    --) shift; break;;
    esac
    shift
done

if which ninja >/dev/null; then
    args+=(-G Ninja)
fi
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
    install_prefix=${OECORE_TARGET_SYSROOT}/install/${build_type}
    args+=(-DCMAKE_TOOLCHAIN_FILE=${OECORE_NATIVE_SYSROOT}/usr/share/cmake/OEToolchainConfig.cmake)
    args+=(-DCMAKE_PREFIX_PATH=/install/${build_type})
fi
args+=(-DCMAKE_INSTALL_PREFIX=${install_prefix:="${install_prefix_default}"})

src_dir_default=$script_path
[ -z ${src_dir:+x} ] && src_dir=${src_dir_default}
# set build dir
build_dir_default=$HOME/build/build.${target_info}/${project_name}
[ -z ${build_dir:+x} ] && build_dir=${build_dir_default}

if [ x${clean:=false} == x"true" ] && [ -d ${build_dir} ];then
    echo "cleaning: rm -fr ${build_dir}"
    rm -fr "${build_dir}"
fi

mkdir -p ${build_dir}
cd -P ${build_dir}
echo "cd $build_dir"
args+=(-B "$build_dir" -S "$src_dir")
echo cmake "${args[@]}"
cmake "${args[@]}"
cmake --build . -j $(nproc)
${build_only:=false} || cmake --install .
${build_package:=false} && cpack -G ${cpack_generator}
if [ -f compile_commands.json ]; then
    cp -av compile_commands.json "$script_path"
fi

exit 0
