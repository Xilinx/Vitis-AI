#
# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

set(sample_name "sample_pointpillars_graph_runner")
set (CMAKE_CXX_STANDARD 17)
find_package(xir)
find_package(Eigen3 REQUIRED)
find_package(OpenCV )
find_package(glog)

if(CPACK_DEBIAN_PACKAGE_ARCHITECTURE STREQUAL "arm64")
  add_compile_options(-Wno-error=attributes -Wno-strict-aliasing)
endif()

add_compile_options( -Wno-array-bounds )

add_executable(${sample_name}
   src/anchor.cpp
   src/helper.cpp
   src/preprocess.cpp
   src/pointpillars_post.cpp
   src/parse_display_result.cpp
   src/main.cpp
)
target_include_directories(${sample_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(${sample_name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(${sample_name} PUBLIC ${OpenCV_LIBS}
                      glog::glog Eigen3::Eigen   vitis_ai_library-graph_runner  vart-runner xir )

