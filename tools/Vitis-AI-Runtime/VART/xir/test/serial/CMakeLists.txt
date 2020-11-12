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

#set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR})

#message(STATUS "PROJECT_SOURCE_DIR = ${PROJECT_SOURCE_DIR}")
#message(STATUS "PROJECT_BINARY_DIR = ${PROJECT_BINARY_DIR}")
#message(STATUS "CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR}")
#message(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")
#message(STATUS "CMAKE_CURRENT_BINARY_DIR = ${CMAKE_CURRENT_BINARY_DIR}")
#message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
#message(STATUS "RUNTIME_OUTPUT_DIRECTORY = ${RUNTIME_OUTPUT_DIRECTORY}")
#message(STATUS "CMAKE_RUNTIME_OUTPUT_DIRECTORY = ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

set(XIR_LIB ${PROJECT_NAME})
include_directories(${CMAKE_BINARY_DIR}/src/)

# test serialize
set(SERIAL_TAR serial.bin)
set(SERIAL_SRC test_serial.cpp resnet.cpp)
add_executable(${SERIAL_TAR} ${SERIAL_SRC})
target_link_libraries(${SERIAL_TAR} glog::glog ${XIR_LIB})

# test de-serialize
set(DESERIAL_TAR deserial.bin)
set(DESERIAL_SRC test_deserial.cpp resnet.cpp)
add_executable(${DESERIAL_TAR} ${DESERIAL_SRC})
target_link_libraries(${DESERIAL_TAR}  glog::glog ${XIR_LIB} )

# test xir_load_and_save
add_executable(xir_load_and_save xir_load_and_save.cpp)
target_link_libraries(xir_load_and_save ${XIR_LIB})
