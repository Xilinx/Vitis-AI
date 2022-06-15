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
  SRCS include/vitis/ai/facedetectrecog.hpp src/facedetectrecog.cpp
       src/facedetectrecog_imp.hpp src/facedetectrecog_imp.cpp
  PUBLIC_REQUIRE ${PROJECT_NAME}::facedetect ${PROJECT_NAME}::facelandmark
                 ${PROJECT_NAME}::facefeature ${PROJECT_NAME}::math
  PRIVATE_REQUIRE xir::xir)

vai_add_test(test_facedetectrecog)
vai_add_test(test_compare_l20)
vai_add_test(test_facedetectrecog_batch)
