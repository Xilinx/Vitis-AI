#
# Copyright 2022-2023 Advanced Micro Devices Inc.
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
  SRCS include/vitis/ai/facerecog.hpp src/facerecog.cpp src/facerecog_imp.hpp
       src/facerecog_imp.cpp src/util.hpp src/util.cpp
  PUBLIC_REQUIRE xir::xir ${PROJECT_NAME}::facefeature
                 ${PROJECT_NAME}::facelandmark)

vai_add_test(test_facerecog REQUIRE ${PROJECT_NAME}::facedetect)
vai_add_test(test_det_facerecog REQUIRE ${PROJECT_NAME}::facedetect)
vai_add_test(test_det_facerecog_batch REQUIRE ${PROJECT_NAME}::facedetect)
