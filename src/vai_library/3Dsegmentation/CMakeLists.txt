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

vai_add_library(
  SRCS include/vitis/ai/3Dsegmentation.hpp src/3Dsegmentation.cpp
       src/3Dsegmentation_imp.hpp src/3Dsegmentation_imp.cpp
  PUBLIC_REQUIRE ${PROJECT_NAME}::xnnpp ${PROJECT_NAME}::dpu_task)


target_include_directories(${COMPONENT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(${COMPONENT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
vai_add_test(test_3Dsegmentation)
vai_add_test(test_3Dsegmentation_batch)
vai_add_test(test_3Dsegmentation_accuracy)
