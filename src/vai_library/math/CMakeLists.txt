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

set(LIBRARY_SOURCE
    src/exp_neon.c
    src/softmax.cpp
    src/ssd_normalizer_c.cpp
    src/ssd_normalizer_neon.cpp
    src/max_index.cpp
    src/yuv2bgr.cpp
    src/tiling.cpp
    # include/vitis/fp16_base.hpp include/vitis/softmax.hpp
    include/vitis/ai/ssd_normalizer.hpp
    include/vitis/ai/math.hpp
    include/vitis/ai/max_index.hpp
    include/vitis/ai/expand_and_align.hpp
    include/vitis/ai/globalavepool.hpp
    include/vitis/ai/image_util.hpp
    include/vitis/ai/preprocess.hpp
    src/expand_and_align.cpp
    src/globalavepool.cpp
    src/image_util.cpp
    src/preprocess_c.cpp
    src/preprocess_neon.cpp
    src/error_code.cpp)

include(CheckIncludeFileCXX)
check_include_file_cxx(arm_neon.h HAVE_NEON)
if(HAVE_NEON)
  list(APPEND LIBRARY_SOURCE src/softmax_table.hpp src/softmax_table.cpp)
endif()

vai_add_library(
  SRCS ${LIBRARY_SOURCE}
  PRIVATE_REQUIRE glog::glog vart::util vart::dpu-controller
  PUBLIC_REQUIRE opencv_core)

if(HAVE_NEON)
  target_compile_definitions(${COMPONENT_NAME} PUBLIC ENABLE_NEON)
endif(HAVE_NEON)

vai_add_test(test_softmax_1 REQUIRE glog::glog)

find_package(GTest REQUIRED)
if(GTest_FOUND)
  vai_add_test(test_normalize REQUIRE GTest::GTest GTest::Main)
endif(GTest_FOUND)

if(HAVE_NEON)
  if(GTest_FOUND)
    vai_add_test(test_max_index REQUIRE GTest::GTest GTest::Main)
  endif(GTest_FOUND)
  vai_add_test(test_softmax_my)
  vai_add_test(bgr_to_yuv420 REQUIRE glog::glog ${OpenCV_LIBS})
  vai_add_test(test_softmax_table)
  vai_add_test(prototype_8bit_to_4bit REQUIRE vart::util)
  vai_add_test(test_transform_mean_scale_performance REQUIRE vart::util)
  vai_add_test(test_transform_mean_scale REQUIRE vart::util)
endif(HAVE_NEON)
