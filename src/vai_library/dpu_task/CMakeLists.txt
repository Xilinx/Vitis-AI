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
  SRCS include/vitis/ai/dpu_task.hpp
       include/vitis/ai/configurable_dpu_task.hpp
       src/dpu_task.cpp
       src/dpu_task_imp.hpp
       src/dpu_task_imp.cpp
       src/banner.hpp
       src/banner.cpp
       src/configurable_dpu_task.cpp
       src/configurable_dpu_task_imp.hpp
       src/configurable_dpu_task_imp.cpp
       src/graph_holder.hpp
       src/graph_holder.cpp
       src/error_code.cpp
  PUBLIC_REQUIRE ${OpenCV_LIBS} vart::util ${PROJECT_NAME}::model_config
                 # for Meta definitions
  PRIVATE_REQUIRE glog::glog xir::xir vart::dpu-runner vart::runner-assistant
                  ${PROJECT_NAME}::graph_runner ${PROJECT_NAME}::math) 

vai_add_test(test_it REQUIRE vart::dpu-runner)
vai_add_test(test_base)
vai_add_test(test_dpu_task)
