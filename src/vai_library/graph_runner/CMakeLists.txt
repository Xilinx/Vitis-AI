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
  src/error_code.cpp
  # public interface
  include/vitis/ai/graph_runner.hpp
  include/vitis/ai/graph_runner.h
  src/graph_runner.cpp
  PUBLIC_REQUIRE
  vart::runner
  ${PROJECT_NAME}::runner_helper
  ${PROJECT_NAME}::cxir
  ${PROJECT_NAME}::cvart
  vart::runner-assistant
  dl)

target_compile_definitions(
  ${COMPONENT_NAME}
  PRIVATE -DGRAPH_RUNNER="$<TARGET_FILE_NAME:${COMPONENT_NAME}>"
          -DCPU_TASK="$<TARGET_FILE_NAME:${PROJECT_NAME}::cpu_task>")
vai_add_test(test_graph_runner)
vai_add_sample(graph_runner_ex_0 DESTINATION graph_runner_ex_0 SRCS
               samples/graph_runner_ex_0.cpp)
vai_add_sample(
  platenum_graph_runner
  DESTINATION
  platenum_graph_runner
  SRCS
  samples/platenum_graph_runner/platenum_graph_runner.cpp
  samples/platenum_graph_runner/readme
  BUILD
  samples/platenum_graph_runner/build.sh
  REQUIRE
  glog::glog
  ${OpenCV_LIBS})
vai_add_sample(
  resnet_v1_50_tf_graph_runner
  DESTINATION
  resnet_v1_50_tf_graph_runner
  SRCS
  samples/resnet_v1_50_tf_graph_runner/resnet_v1_50_tf_graph_runner.cpp
  samples/resnet_v1_50_tf_graph_runner/word_list.inc
  samples/resnet_v1_50_tf_graph_runner/readme
  BUILD
  samples/resnet_v1_50_tf_graph_runner/build.sh
  REQUIRE
  glog::glog
  ${OpenCV_LIBS})

vai_add_sample(
  yolov2_regionop_graph_runner
  DESTINATION
  yolov2_regionop_graph_runner
  SRCS
  samples/yolov2_regionop_graph_runner/yolov2_regionop_graph_runner.cpp
  samples/yolov2_regionop_graph_runner/readme
  BUILD
  samples/yolov2_regionop_graph_runner/build.sh
  REQUIRE
  glog::glog
  ${OpenCV_LIBS})

vai_add_sample(
  tfssd_gridanchor_nms_op_graph_runner
  DESTINATION
  tfssd_gridanchor_nms_op_graph_runner
  SRCS
  samples/tfssd_gridanchor_nms_op_graph_runner/tfssd_gridanchor_nms_op_graph_runner.cpp
  samples/tfssd_gridanchor_nms_op_graph_runner/readme
  BUILD
  samples/tfssd_gridanchor_nms_op_graph_runner/build.sh
  REQUIRE
  glog::glog
  ${OpenCV_LIBS})

# comment out for docker build temporarily add_subdirectory(
# samples/pointpillars_graph_runner) begin "sample_pointpillars_graph_runner"
# set(pp_sample_name "sample_pointpillars_graph_runner") set(PBF_NAME2
# activations anchors box_coder input_reader layers losses model optimizer
# pipeline preprocess sampler second similarity target train voxel_generator)
# set (CMAKE_CXX_STANDARD 17) find_package(Eigen3 REQUIRED)
#
# foreach(PBF ${PBF_NAME2}) add_custom_command( OUTPUT
# ${CMAKE_CURRENT_BINARY_DIR}/second/protos/${PBF}.pb.cc
# ${CMAKE_CURRENT_BINARY_DIR}/second/protos/${PBF}.pb.h DEPENDS
# ${CMAKE_CURRENT_SOURCE_DIR}/samples/pointpillars_graph_runner/include/second/protos/${PBF}.proto
# COMMAND protoc ARGS --cpp_out ${CMAKE_CURRENT_BINARY_DIR} --proto_path
# ${CMAKE_CURRENT_SOURCE_DIR}/samples/pointpillars_graph_runner/include
# ${CMAKE_CURRENT_SOURCE_DIR}/samples/pointpillars_graph_runner/include/second/protos/${PBF}.proto
# COMMENT "Running C++ protocol buffer compiler on proto ${PBF}" VERBATIM)
# list(APPEND LIST_PBSRC_CC
# ${CMAKE_CURRENT_BINARY_DIR}/second/protos/${PBF}.pb.cc) list(APPEND
# LIST_PBSRC_H2 ${CMAKE_CURRENT_BINARY_DIR}/second/protos/${PBF}.pb.h)
# endforeach()
#
# if(CPACK_DEBIAN_PACKAGE_ARCHITECTURE STREQUAL "arm64")
# add_compile_options(-Wno-error=attributes -Wno-strict-aliasing) endif()
#
# add_compile_options( -Wno-array-bounds ) add_executable(${pp_sample_name}
# samples/pointpillars_graph_runner/src/anchor.cpp
# samples/pointpillars_graph_runner/src/helper.cpp
# samples/pointpillars_graph_runner/src/preprocess.cpp
# samples/pointpillars_graph_runner/src/pointpillars_post.cpp
# samples/pointpillars_graph_runner/src/parse_display_result.cpp
# samples/pointpillars_graph_runner/src/main.cpp ${LIST_PBSRC_CC} )
#
# target_include_directories(${pp_sample_name} PRIVATE
# ${CMAKE_CURRENT_SOURCE_DIR}/samples/pointpillars_graph_runner/src)
# target_include_directories(${pp_sample_name} PRIVATE
# ${CMAKE_CURRENT_BINARY_DIR}/samples/pointpillars_graph_runner/)
# target_include_directories(${pp_sample_name} PRIVATE
# ${CMAKE_CURRENT_SOURCE_DIR}/include)
# target_include_directories(${pp_sample_name} PRIVATE
# ${CMAKE_CURRENT_BINARY_DIR}/) target_link_libraries(${pp_sample_name} PUBLIC
# protobuf Eigen3::Eigen ${OpenCV_LIBS}  ${PROJECT_NAME}::${COMPONENT_NAME}  )
#
# install( DIRECTORY samples/pointpillars_graph_runner DESTINATION
# ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/samples/${COMPONENT_NAME} )
# install( TARGETS sample_pointpillars_graph_runner DESTINATION
# ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/samples/${COMPONENT_NAME}/pointpillars_graph_runner
# ) end "sample_pointpillars_graph_runner"

install(
  FILES samples/resnet_v1_50_tf_graph_runner_py/resnet_v1_50_tf.py
        samples/resnet_v1_50_tf_graph_runner_py/words.txt
  DESTINATION
    ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/samples/${COMPONENT_NAME}/resnet_v1_50_tf_graph_runner_py
)

if(BUILD_PYTHON)
  vai_add_pybind11_module(py_graph_runner MODULE_NAME vitis_ai_library
                          python/graph_runner_py_module.cpp)
  target_link_libraries(py_graph_runner
                        PRIVATE ${PROJECT_NAME}::${COMPONENT_NAME})
endif(BUILD_PYTHON)
