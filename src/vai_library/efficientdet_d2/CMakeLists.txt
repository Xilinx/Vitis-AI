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
  SRCS include/vitis/ai/efficientdet_d2.hpp src/efficientdet_d2.cpp 
       src/efficientdet_d2_imp.cpp src/efficientdet_d2_imp.hpp 
       #src/anchor.hpp src/anchor.cpp 
       #src/object_detection_base.hpp src/object_detection_base.cpp 
       #src/postprocess.hpp src/postprocess.cpp
  PUBLIC_REQUIRE ${PROJECT_NAME}::xnnpp ${PROJECT_NAME}::dpu_task)

vai_add_test(test_efficientdet_d2)
vai_add_test(test_efficientdet_d2_batch)
