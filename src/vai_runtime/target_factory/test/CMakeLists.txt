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

include_directories(${CMAKE_BINARY_DIR}/src)

add_executable(test_factory test_factory.cpp)
target_link_libraries(test_factory ${PROJECT_NAME} protobuf::libprotobuf unilog::unilog)

add_executable(test_fingerprint test_fingerprint.cpp)
target_link_libraries(test_fingerprint ${PROJECT_NAME} protobuf::libprotobuf unilog::unilog)

add_executable(test_lib_info test_lib_info.cpp)
target_link_libraries(test_lib_info ${PROJECT_NAME} unilog::unilog)
