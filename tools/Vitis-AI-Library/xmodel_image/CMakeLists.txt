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

# add_custom_command( OUTPUT
# ${CMAKE_CURRENT_BINARY_DIR}/vitis/ai/xmodel_image.pb.cc
# ${CMAKE_CURRENT_BINARY_DIR}/vitis/ai/xmodel_image.pb.h DEPENDS
# ${CMAKE_CURRENT_SOURCE_DIR}/include/vitis/ai/xmodel_image.proto COMMAND
# protobuf::protoc ARGS --cpp_out ${CMAKE_CURRENT_BINARY_DIR} --proto_path
# ${CMAKE_CURRENT_SOURCE_DIR}/include
# ${CMAKE_CURRENT_SOURCE_DIR}/include/vitis/ai/xmodel_image.proto COMMENT
# "Running C++ protocol buffer compiler on proto xmodel_image.proto" VERBATIM)

vai_add_library(
  SRCS ./include/vitis/ai/xmodel_image.hpp
       ./include/vitis/ai/xmodel_preprocessor.hpp
       ./include/vitis/ai/xmodel_postprocessor.hpp
       # ${CMAKE_CURRENT_BINARY_DIR}/vitis/ai/xmodel_image.pb.cc
       # ${CMAKE_CURRENT_BINARY_DIR}/vitis/ai/xmodel_image.pb.h
       src/xmodel_image.cpp
       src/xmodel_image_imp.hpp
       src/xmodel_image_imp.cpp
       src/graph_holder.hpp
       src/graph_holder.cpp
       src/xmodel_preprocessor.cpp
       src/xmodel_postprocessor.cpp
  PUBLIC_REQUIRE ${PROJECT_NAME}::model_config
  PRIVATE_REQUIRE
    ${PROJECT_NAME}::math
    ${PROJECT_NAME}::xmodel_jit
    vart::runner
    ${PROJECT_NAME}::runner_helper
    xir::xir
    glog::glog
    dl)

target_include_directories(
  ${COMPONENT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
                           $<INSTALL_INTERFACE:include>)

# install(FILES ./include/vitis/ai/xmodel_image.proto #
# ${CMAKE_CURRENT_BINARY_DIR}/vitis/ai/xmodel_image.pb.h DESTINATION
# include/vitis/ai/)

vai_add_sample(
  xmodel_example_0
  SRCS samples/xmodel_example_0.cpp samples/process_image.hpp
       samples/xmodel_result_to_string.hpp
  REQUIRE glog::glog ${OpenCV_LIBS})

vai_add_test(
  test_postprocessor
  REQUIRE ${PROJECT_NAME}::xmodel_jit ${PROJECT_NAME}::runner_helper
          vart::runner-assistant glog::glog)

macro(ADD_PREPROCESSOR NAME)
  add_library(${NAME}_preprocessor_target SHARED ${ARGN})
  target_link_libraries(
    ${NAME}_preprocessor_target
    PRIVATE ${COMPONENT_NAME} vart::runner xir::xir
            ${PROJECT_NAME}::runner_helper ${OpenCV_LIBS})
  target_include_directories(
    ${NAME}_preprocessor_target
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>)
  set_target_properties(
    ${NAME}_preprocessor_target
    PROPERTIES VERSION "${PROJECT_VERSION}"
               SOVERSION "${PROJECT_VERSION_MAJOR}"
               LIBRARY_OUTPUT_NAME xmodel_preprocessor_${NAME})
  install(
    TARGETS ${NAME}_preprocessor_target
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib)
endmacro()

add_preprocessor(common preprocessors/common/common.cpp)
add_preprocessor(vgg preprocessors/vgg/vgg_preprocessor.cpp)
add_preprocessor(efficientnet
                 preprocessors/efficientnet/efficientnet_preprocessor.cpp)

macro(ADD_POSTPROCESSOR NAME)
  add_library(${NAME}_postprocessor_target SHARED ${ARGN})
  target_link_libraries(
    ${NAME}_postprocessor_target
    PRIVATE ${COMPONENT_NAME} vart::runner xir::xir
            ${PROJECT_NAME}::runner_helper ${OpenCV_LIBS})
  target_include_directories(
    ${NAME}_postprocessor_target
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>)
  set_target_properties(
    ${NAME}_postprocessor_target
    PROPERTIES VERSION "${PROJECT_VERSION}"
               SOVERSION "${PROJECT_VERSION_MAJOR}"
               LIBRARY_OUTPUT_NAME xmodel_postprocessor_${NAME})
  install(
    TARGETS ${NAME}_postprocessor_target
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib)
endmacro()

add_postprocessor(common postprocessors/common/common.cpp)
add_postprocessor(classification
                  postprocessors/classification/classification.cpp)
add_postprocessor(densebox postprocessors/densebox/densebox.cpp)
add_postprocessor(plate_number postprocessors/plate_number/plate_number.cpp)
add_postprocessor(face_recognition
                  postprocessors/face_recognition/face_recognition.cpp)
add_postprocessor(face_quality postprocessors/face_quality/face_quality.cpp)
add_postprocessor(face_landmark postprocessors/face_landmark/face_landmark.cpp)
add_postprocessor(
  lane_detect postprocessors/lane_detect/lane_detect.cpp
  postprocessors/lane_detect/ipm_info.hpp
  postprocessors/lane_detect/ipm_info.cpp)
add_postprocessor(open_pose postprocessors/open_pose/open_pose.cpp)
add_postprocessor(segmentation postprocessors/segmentation/segmentation.cpp)

if(BUILD_PYTHON)
  vai_add_pybind11_module(xmodel_image_py_module MODULE_NAME xmodel_image
                          PACKAGE_NAME vitis.ai src/xmodel_image_py_module.cpp)
  target_link_libraries(xmodel_image_py_module PRIVATE ${COMPONENT_NAME}
                                                       glog::glog)
  add_custom_target(
    py_xmodel_image_proto
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/vitis/ai/xmodel_image.proto
    COMMAND
      protobuf::protoc --python_out ${CMAKE_CURRENT_BINARY_DIR} --proto_path
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/include/vitis/ai/xmodel_image.proto
    COMMENT
      "Running PYTHON protocol buffer compiler on proto xmodel_image.proto"
    VERBATIM)
  add_dependencies(xmodel_image_py_module py_xmodel_image_proto)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/vitis/ai/xmodel_image_pb2.py
          DESTINATION ${PYTHON_SITE_PACKAGES}/vitis/ai/)
endif(BUILD_PYTHON)
