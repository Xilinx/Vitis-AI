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
  SRCS # internal implementation the end user does not need to access GraphTask
       src/graph_runner.hpp
       src/graph_runner.cpp
       src/graph_runner_imp.cpp
       src/tensor_buffer_linker.hpp
       src/tensor_buffer_linker.cpp
       src/tensor_buffer_linker_v.hpp
       src/tensor_buffer_linker_v.cpp
       src/tensor_buffer_linker_p.hpp
       src/tensor_buffer_linker_p.cpp
       src/tensor_buffer_shared.hpp
       src/tensor_buffer_shared.cpp
       # public interface
       include/vitis/ai/graph_runner.hpp
       include/vitis/ai/graph_runner.h
       src/graph_runner.cpp
  PUBLIC_REQUIRE vart::runner ${PROJECT_NAME}::runner_helper
                 vart::runner-assistant dl)

target_compile_definitions(
  ${COMPONENT_NAME}
  PRIVATE -DGRAPH_RUNNER="$<TARGET_FILE_NAME:${COMPONENT_NAME}>"
          -DCPU_TASK="$<TARGET_FILE_NAME:${PROJECT_NAME}::cpu_task>")
vai_add_test(test_graph_runner)
vai_add_graphrunner_sample(graph_runner_ex_0 graph_runner_ex_0 SRCS samples/graph_runner_ex_0.cpp)
vai_add_graphrunner_sample(
  platenum_graph_runner
  platenum_graph_runner
  SRCS samples/platenum_graph_runner/platenum_graph_runner.cpp
       samples/platenum_graph_runner/build.sh
       samples/platenum_graph_runner/readme
  REQUIRE glog::glog ${OpenCV_LIBS})
vai_add_graphrunner_sample(
  resnet50_graph_runner
  resnet50_graph_runner
  SRCS samples/resnet50_graph_runner/resnet50_graph_runner.cpp
       samples/resnet50_graph_runner/word_list.inc
       samples/resnet50_graph_runner/build.sh
       samples/resnet50_graph_runner/readme
  REQUIRE glog::glog ${OpenCV_LIBS})
vai_add_graphrunner_sample(
  resnet_v1_50_tf_graph_runner
  resnet_v1_50_tf_graph_runner
  SRCS samples/resnet_v1_50_tf_graph_runner/resnet_v1_50_tf_graph_runner.cpp
       samples/resnet_v1_50_tf_graph_runner/word_list.inc
       samples/resnet_v1_50_tf_graph_runner/build.sh
       samples/resnet_v1_50_tf_graph_runner/readme
  REQUIRE glog::glog ${OpenCV_LIBS})

install(
  FILES samples/resnet50_graph_runner_py/resnet50.py
        samples/resnet50_graph_runner_py/words.txt
  DESTINATION
    ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/samples/${COMPONENT_NAME}/resnet50_graph_runner_py)

if(BUILD_PYTHON)
  vai_add_pybind11_module(py_graph_runner MODULE_NAME vitis_ai_library
                          python/graph_runner_py_module.cpp)
  target_link_libraries(py_graph_runner
                        PRIVATE ${PROJECT_NAME}::${COMPONENT_NAME})
endif(BUILD_PYTHON)
