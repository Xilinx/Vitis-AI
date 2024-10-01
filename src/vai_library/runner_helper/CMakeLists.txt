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
  SRCS ./src/runner_helper.cpp
       ./include/vart/runner_helper.hpp
       ./include/vitis/ai/path_util.hpp
       src/path_util.cpp
       ./include/vart/batch_tensor_buffer_view.hpp
       src/batch_tensor_buffer_view.cpp
       src/tensor_mirror_attrs.hpp
       src/tensor_mirror_attrs.cpp
  PUBLIC_REQUIRE vart::runner vart::util glog::glog)

vai_add_test(test_path_util)
