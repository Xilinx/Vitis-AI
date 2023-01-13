set -e
set -x
BUILD_VER=${BUILD_VER:-'dev'}
declare -a args
pwd=$(realpath $PWD)
args+=(--cmake-options=-Donnxruntime_RUN_ONNX_TESTS=OFF)
args+=(--cmake-options=-Donnxruntime_BUILD_WINML_TESTS=ON)
args+=(--cmake-options=-Donnxruntime_GENERATE_TEST_REPORTS=ON)
args+=(--cmake-options=-Donnxruntime_ROCM_VERSION=)
args+=(--cmake-options=-Donnxruntime_USE_MIMALLOC=OFF)
args+=(--cmake-options=-Donnxruntime_BUILD_CSHARP=OFF)
args+=(--cmake-options=-Donnxruntime_BUILD_JAVA=OFF)
args+=(--cmake-options=-Donnxruntime_BUILD_NODEJS=OFF)
args+=(--cmake-options=-Donnxruntime_BUILD_OBJC=OFF)
args+=(--cmake-options=-Donnxruntime_BUILD_APPLE_FRAMEWORK=OFF)
args+=(--cmake-options=-Donnxruntime_USE_DNNL=OFF)
args+=(--cmake-options=-Donnxruntime_DNNL_GPU_RUNTIME=)
args+=(--cmake-options=-Donnxruntime_DNNL_OPENCL_ROOT=)
args+=(--cmake-options=-Donnxruntime_USE_NNAPI_BUILTIN=OFF)
args+=(--cmake-options=-Donnxruntime_USE_RKNPU=OFF)
args+=(--cmake-options=-Donnxruntime_USE_OPENMP=OFF)
args+=(--cmake-options=-Donnxruntime_USE_NUPHAR_TVM=OFF)
args+=(--cmake-options=-Donnxruntime_USE_LLVM=OFF)
args+=(--cmake-options=-Donnxruntime_ENABLE_MICROSOFT_INTERNAL=OFF)
args+=(--cmake-options=-Donnxruntime_USE_VITISAI=ON)
args+=(--cmake-options=-Donnxruntime_USE_NUPHAR=OFF)
args+=(--cmake-options=-Donnxruntime_USE_TENSORRT=OFF)
args+=(--cmake-options=-Donnxruntime_TENSORRT_HOME=)
args+=(--cmake-options=-Donnxruntime_USE_TVM=OFF)
args+=(--cmake-options=-Donnxruntime_TVM_CUDA_RUNTIME=OFF)
args+=(--cmake-options=-Donnxruntime_USE_MIGRAPHX=OFF)
args+=(--cmake-options=-Donnxruntime_MIGRAPHX_HOME=)
args+=(--cmake-options=-Donnxruntime_CROSS_COMPILING=OFF)
args+=(--cmake-options=-Donnxruntime_DISABLE_CONTRIB_OPS=OFF)
args+=(--cmake-options=-Donnxruntime_DISABLE_ML_OPS=OFF)
args+=(--cmake-options=-Donnxruntime_DISABLE_RTTI=OFF)
args+=(--cmake-options=-Donnxruntime_DISABLE_EXCEPTIONS=OFF)
args+=(--cmake-options=-Donnxruntime_MINIMAL_BUILD=OFF)
args+=(--cmake-options=-Donnxruntime_EXTENDED_MINIMAL_BUILD=OFF)
args+=(--cmake-options=-Donnxruntime_MINIMAL_BUILD_CUSTOM_OPS=OFF)
args+=(--cmake-options=-Donnxruntime_REDUCED_OPS_BUILD=OFF)
args+=(--cmake-options=-Donnxruntime_ENABLE_LANGUAGE_INTEROP_OPS=OFF)
args+=(--cmake-options=-Donnxruntime_USE_DML=OFF)
args+=(--cmake-options=-Donnxruntime_USE_WINML=OFF)
args+=(--cmake-options=-Donnxruntime_BUILD_MS_EXPERIMENTAL_OPS=OFF)
args+=(--cmake-options=-Donnxruntime_USE_TELEMETRY=OFF)
args+=(--cmake-options=-Donnxruntime_ENABLE_LTO=OFF)
args+=(--cmake-options=-Donnxruntime_ENABLE_TRANSFORMERS_TOOL_TEST=OFF)
args+=(--cmake-options=-Donnxruntime_USE_ACL=OFF)
args+=(--cmake-options=-Donnxruntime_USE_ACL_1902=OFF)
args+=(--cmake-options=-Donnxruntime_USE_ACL_1905=OFF)
args+=(--cmake-options=-Donnxruntime_USE_ACL_1908=OFF)
args+=(--cmake-options=-Donnxruntime_USE_ACL_2002=OFF)
args+=(--cmake-options=-Donnxruntime_USE_ARMNN=OFF)
args+=(--cmake-options=-Donnxruntime_ARMNN_RELU_USE_CPU=ON)
args+=(--cmake-options=-Donnxruntime_ARMNN_BN_USE_CPU=ON)
args+=(--cmake-options=-Donnxruntime_ENABLE_NVTX_PROFILE=OFF)
args+=(--cmake-options=-Donnxruntime_ENABLE_TRAINING=OFF)
args+=(--cmake-options=-Donnxruntime_ENABLE_TRAINING_OPS=OFF)
args+=(--cmake-options=-Donnxruntime_ENABLE_TRAINING_TORCH_INTEROP=OFF)
args+=(--cmake-options=-Donnxruntime_ENABLE_CPU_FP16_OPS=OFF)
args+=(--cmake-options=-Donnxruntime_USE_NCCL=ON)
args+=(--cmake-options=-Donnxruntime_BUILD_BENCHMARKS=OFF)
args+=(--cmake-options=-Donnxruntime_USE_ROCM=OFF)
args+=(--cmake-options=-Donnxruntime_ROCM_HOME=)
args+=(--cmake-options=-DOnnxruntime_GCOV_COVERAGE=OFF)
args+=(--cmake-options=-Donnxruntime_USE_MPI=ON)
args+=(--cmake-options=-Donnxruntime_ENABLE_MEMORY_PROFILE=OFF)
args+=(--cmake-options=-Donnxruntime_ENABLE_CUDA_LINE_NUMBER_INFO=OFF)
args+=(--cmake-options=-Donnxruntime_BUILD_WEBASSEMBLY=OFF)
args+=(--cmake-options=-Donnxruntime_BUILD_WEBASSEMBLY_STATIC_LIB=OFF)
args+=(--cmake-options=-Donnxruntime_ENABLE_WEBASSEMBLY_SIMD=OFF)
args+=(--cmake-options=-Donnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_CATCHING=ON)
args+=(--cmake-options=-Donnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_THROWING=OFF)
args+=(--cmake-options=-Donnxruntime_ENABLE_WEBASSEMBLY_THREADS=OFF)
args+=(--cmake-options=-Donnxruntime_ENABLE_WEBASSEMBLY_DEBUG_INFO=OFF)
args+=(--cmake-options=-Donnxruntime_ENABLE_WEBASSEMBLY_PROFILING=OFF)
args+=(--cmake-options=-Donnxruntime_ENABLE_EAGER_MODE=OFF)
args+=(--cmake-options=-Donnxruntime_ENABLE_EXTERNAL_CUSTOM_OP_SCHEMAS=OFF)
args+=(--cmake-options=-Donnxruntime_NVCC_THREADS=0)
args+=(--cmake-options=-Donnxruntime_ENABLE_CUDA_PROFILING=OFF)
args+=(--cmake-options=-Donnxruntime_BUILD_SHARED_LIB=ON)
##BUILD PYTHON RELATE
args+=(--cmake-options=-DBUILD_ONNX_PYTHON=ON)
args+=(--cmake-options=-Donnxruntime_ENABLE_PYTHON=ON)
args+=(--cmake-options=-Donnxruntime_PYBIND_EXPORT_OPSCHEMA=ON)
args+=(--cmake-options=-Donnxruntime_ENABLE_LANGUAGE_INTEROP_OPS=ON)
#args+=(--cmake-options=-DBUILD_ONNX_PYTHON=OFF)
#args+=(--cmake-options=-Donnxruntime_ENABLE_PYTHON=OFF)
#args+=(--cmake-options=-Donnxruntime_PYBIND_EXPORT_OPSCHEMA=OFF)

args+=(--cmake-options=-DProtobuf_USE_STATIC_LIBS=OFF)
args+=(--cmake-options=-Donnxruntime_USE_PREINSTALLED_EIGEN=ON)
args+=(--cmake-options=-DONNX_USE_PROTOBUF_SHARED_LIBS=ON)
args+=(--cmake-options=-Donnxruntime_PREFER_SYSTEM_LIB=ON)
args+=(--cmake-options=-DABSL_ENABLE_INSTALL=ON)
args+=(--cmake-options=-DBUILD_SHARED_LIBS=ON)
args+=(--cmake-options=-Donnxruntime_USE_FULL_PROTOBUF=ON)
args+=(--cmake-options=-Donnxruntime_DEV_MODE=ON)
args+=(--cmake-options=-Donnxruntime_ENABLE_MEMLEAK_CHECKER=ON)
args+=(--cmake-options=-DPY_VERSION=3)
args+=(--cmake-options=-Deigen_SOURCE_PATH=$pwd/../onnxruntime/cmake/external/eigen)
build_type=Debug
if echo $@ | grep release; then
    build_type=Release
fi
cpack_generator=RPM
if echo $@ | grep deb; then
    cpack_generator=DEB
fi

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
    args+=(--cmake-options=-DPython_NumPy_INCLUDE_DIR=${OECORE_NATIVE_SYSROOT}/usr/lib/python3.9/site-packages/numpy/core/include)
    target_info=${OECORE_TARGET_OS}.${OECORE_SDK_VERSION}.${OECORE_TARGET_ARCH}.${build_type}
fi
build_dir=$HOME/build/build.${target_info}/onnxruntime
$pwd/cmake.sh --project onnxruntime --src-dir=$pwd/../onnxruntime/cmake --build-dir $build_dir  "${args[@]}" $@

#package onnxruntime
cpack  -G ${cpack_generator} -B $build_dir -P onnxruntime -R ${BUILD_VER} -D CPACK_CMAKE_GENERATOR=Ninja -D CPACK_INSTALL_CMAKE_PROJECTS="$build_dir;onnxruntime;ALL;/" -D CPACK_PACKAGE_FILE_NAME=libonnxruntime-${BUILD_VER}-aarch64 -D CPACK_PACKAGE_DESCRIPTION_FILE=$pwd/../onnxruntime/README.md 

cmake --build $build_dir --target "external/onnx/all" -j $(nproc)
cmake --install $build_dir/external/onnx
#package onnx
cpack  -G ${cpack_generator} -B $build_dir/external/onnx -P onnx -R ${BUILD_VER} -D CPACK_CMAKE_GENERATOR=Ninja -D CPACK_INSTALL_CMAKE_PROJECTS="$build_dir/external/onnx;onnx;ALL;/" -D CPACK_PACKAGE_FILE_NAME=libonnx-${BUILD_VER}-aarch64 -D CPACK_PACKAGE_DESCRIPTION_FILE=$pwd/../onnxruntime/cmake/external/onnx/README.md



