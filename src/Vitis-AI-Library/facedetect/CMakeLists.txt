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
  SRCS include/vitis/ai/facedetect.hpp src/facedetect.cpp src/detect_imp.hpp
       src/detect_imp.cpp #
  PUBLIC_REQUIRE ${PROJECT_NAME}::xnnpp ${PROJECT_NAME}::dpu_task)

vai_add_test(test_facedetect)
vai_add_test(test_facedetect_batch)
vai_add_test(test_facedetect_accuracy)

if(XRT_FOUND)
  vai_add_test(
    test_facedetect_zero_copy test/test_facedetect_zero_copy.cpp
    # for allocate bo
    REQUIRE XRT::XRT)
endif(XRT_FOUND)
