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
  SRCS include/xilinx/ai/tracker.hpp
       src/ftd/ftd_filter_linear.cpp
       src/ftd/ftd_structure.cpp
       src/ftd/ftd_trajectory.cpp
       src/ftd/ftd_filter_linear.hpp
       src/ftd/ftd_structure.hpp
       src/ftd/ftd_trajectory.hpp
       src/common.hpp
       src/ring_queue.hpp
       src/state_map.cpp
       src/state_map.hpp
       src/tracker.cpp
       src/tracker_imp.cpp
       src/tracker_imp.hpp)

vai_add_test(test_track)
