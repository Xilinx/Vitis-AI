cmake_minimum_required(VERSION 2.6)
project(dpu4rnn)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")
 
LINK_DIRECTORIES(/usr/lib)
LINK_DIRECTORIES(/usr/local/lib)
LINK_DIRECTORIES(/opt/xilinx/xrt/lib)
link_directories(/usr/local/anaconda/envs/pytorch/lib/python3.7/)

INCLUDE_DIRECTORIES(/usr/include)
INCLUDE_DIRECTORIES(/usr/local/include)
INCLUDE_DIRECTORIES(/opt/xilinx/xrt/include)
INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(src)
include_directories(/usr/local/anaconda/envs/pytorch/include/pybind11/)
include_directories(/usr/local/anaconda3/envs/pytorch/include/python3.7m/)

set(LIB_NAME ${PROJECT_NAME})
set(PY_LIB_NAME dpu4rnn_py)

set(DEPS_PUBLIC xir unilog vart-runner vart-xrnn-runner uuid util glog)
set(DEPS_PRIVATE  vart-xrt-device-handle vart-buffer-object)


aux_source_directory(src SOURCE_FILES)
add_library(${LIB_NAME} SHARED  ${SOURCE_FILES})
target_link_libraries(${LIB_NAME} vart-runner xir glog)

add_executable(test ${SOURCE_FILES} test.cpp)
target_link_libraries(test vart-runner xir glog)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

add_library(${PY_LIB_NAME} SHARED python/dpu4rnn_python.cpp ${SOURCE_FILES})
set_target_properties(${PY_LIB_NAME} PROPERTIES PREFIX "")
target_link_libraries(${PY_LIB_NAME} vart-runner xir glog)

