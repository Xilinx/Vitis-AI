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
# ==============================================================================

cmake_minimum_required(VERSION 3.5)

project(vai_q_tensorflow VERSION 2.0)

configure_file(VAI_QConfig.h.in VAI_QConfig.h)

message("PROJECT_SOURCE_DIR is : ${PROJECT_SOURCE_DIR}")

execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()), end='')" OUTPUT_VARIABLE Tensorflow_LINK_FLAGS)
option(ABI1 "option for buidl with ABI=1" ON)
if (ABI1)
  message("build with ABI=1")
  execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()).replace('ABI=0', 'ABI=1'), end='')" OUTPUT_VARIABLE Tensorflow_COMPILE_FLAGS)
else()
  message("build with ABI=0")
  execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()).replace('ABI=0', 'ABI=0'), end='')" OUTPUT_VARIABLE Tensorflow_COMPILE_FLAGS)
endif()
execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_lib())" OUTPUT_VARIABLE TF_LIBRARY_PATH)

string(STRIP "${Tensorflow_INCLUDE_DIRS}" Tensorflow_INCLUDE_DIRS)
string(STRIP "${TF_LIBRARY_PATH}" TF_LIBRARY_PATH)

# clear CMAKE_CXX_FLAGS
set(CMAKE_CXX_FLAGS "")
message("Init CMAKE_CXX_FLAGS is  ${CMAKE_CXX_FLAGS}")

# specify the C++ standard
# set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)


FIND_PACKAGE(PythonInterp)
FIND_PACKAGE(PythonLibs)

find_package (Python3 COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})
link_libraries(${Python3_LIBRARIES})

message("CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
include_directories("${CMAKE_INSTALL_PREFIX}/include")
link_directories("${CMAKE_INSTALL_PREFIX}/lib")


find_package(SWIG COMPONENTS python)
INCLUDE(${SWIG_USE_FILE})
if(SWIG_FOUND)
  message("SWIG found: ${SWIG_EXECUTABLE}")
endif()



# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-mno-avx512f -march=native -Wno-sign-compare -std=c++14 ${Tensorflow_COMPILE_FLAGS} -I${CMAKE_INSTALL_PREFIX}/include ${CMAKE_CXX_FLAGS}")

option(DEBUG "option for DEBUG" OFF)
if (DEBUG)
  message("build with debug mode")
  set(CMAKE_CXX_FLAGS "-O0 -ggdb ${CMAKE_CXX_FLAGS}")
endif()
## debug mode

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed -Wl,-rpath='$ORIGIN/':'$ORIGIN/../../tensorflow_core/' ${Tensorflow_LINK_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")

add_subdirectory(fix_neuron_op)


include_directories("./include")

link_directories(${TF_LIBRARY_PATH})

message("vai_q_tensorflow build")
message("tensorflow library path dir is  ${TF_LIBRARY_PATH}")
message("tensorflow include dir is  ${Tensorflow_INCLUDE_DIRS}")
message("tensorflow link flag is  ${Tensorflow_LINK_FLAGS}")
message("tensorflow compile flag is  ${Tensorflow_COMPILE_FLAGS}")
message("PYTHON_INCLUDE_PATH: ${PYTHON_INCLUDE_PATH}")

message("CMAKE_CXX_FLAGS is  ${CMAKE_CXX_FLAGS}")
message("CMAKE_SHARED_LINKER_FLAGS is  ${CMAKE_SHARED_LINKER_FLAGS}")

aux_source_directory(src SRC_LIST)

#Python file output directory - outdir
set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_BINARY_DIR})

SET(CMAKE_SWIG_FLAGS "")

set(SWIG_FILES ${CMAKE_CURRENT_SOURCE_DIR}/swig/vai_wrapper.i)
set_property(SOURCE ${SWIG_FILES} PROPERTY CPLUSPLUS ON)

SET_SOURCE_FILES_PROPERTIES(${SWIG_FILES} PROPERTIES SWIG_FLAGS "-includeall")

set(SWIG_SRC "${SWIG_FILES};${SRC_LIST}")
message("SWIG_SRC is : ${SWIG_SRC}")

SWIG_ADD_LIBRARY(vai_wrapper LANGUAGE python SOURCES ${SWIG_SRC})
SWIG_LINK_LIBRARIES(vai_wrapper PUBLIC ${PYTHON_LIBRARIES}
  ${Tensorflow_LINK_FLAGS})
