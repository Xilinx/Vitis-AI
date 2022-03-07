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
find_library(LIB_JSON_C json-c)

vai_add_library(
  SRCS
  ./include/vitis/ai/xmodel_image.hpp
  ./include/vitis/ai/xmodel_preprocessor.hpp
  ./include/vitis/ai/xmodel_postprocessor.hpp
  ./include/vitis/ai/json_object_visitor.hpp
  src/xmodel_image.cpp
  src/xmodel_image_imp.hpp
  src/xmodel_image_imp.cpp
  src/graph_holder.hpp
  src/graph_holder.cpp
  src/xmodel_preprocessor.cpp
  src/xmodel_postprocessor.cpp
  PUBLIC_REQUIRE
  ${PROJECT_NAME}::model_config
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

vai_add_sample(
  xmodel_example_0
  SRCS
  samples/xmodel_example_0.cpp
  samples/process_image.hpp
  samples/xmodel_result_to_string.hpp
  REQUIRE
  glog::glog
  ${OpenCV_LIBS})

vai_add_sample(
  test_performance_xmodel_image
  SRCS
  samples/test_performance_xmodel_image.cpp
  samples/benchmark.hpp
  REQUIRE
  glog::glog
  ${OpenCV_LIBS}
  vart::util
  ${PROJECT_NAME}::benchmark
  pthread)

vai_add_sample(test_jpeg_xmodel_image SRCS samples/test_jpeg_xmodel_image.cpp
               REQUIRE glog::glog ${OpenCV_LIBS})

vai_add_test(test_postprocessor REQUIRE ${PROJECT_NAME}::xmodel_jit
             ${PROJECT_NAME}::runner_helper vart::runner-assistant glog::glog)

if(LIB_JSON_C)
  vai_add_test(test_json_object_visitor REQUIRE ${LIB_JSON_C} glog::glog)
else(LIB_JSON_C)
  message(WARNING "json-c not found, test_json_object_visitor need json-c")
endif(LIB_JSON_C)

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

add_postprocessor(yolov3 postprocessors/yolov3/yolov3.cpp)
target_link_libraries(yolov3_postprocessor_target
                      PRIVATE ${PROJECT_NAME}::xnnpp)

if(LIB_JSON_C)
  add_postprocessor(
    ssd postprocessors/ssd/ssd.cpp postprocessors/ssd/prior_boxes.hpp
    postprocessors/ssd/prior_boxes.cpp postprocessors/ssd/ssd_detector.hpp
    postprocessors/ssd/ssd_detector.cpp)
  target_link_libraries(ssd_postprocessor_target PRIVATE ${PROJECT_NAME}::xnnpp
                                                         ${LIB_JSON_C})
else(LIB_JSON_C)
  message(WARNING "json-c not found, ssd postprocessor need json-c")
endif(LIB_JSON_C)

if(BUILD_PYTHON)
  vai_add_pybind11_module(xmodel_image_py_module MODULE_NAME xmodel_image
                          PACKAGE_NAME vitis.ai src/xmodel_image_py_module.cpp)
  target_link_libraries(xmodel_image_py_module PRIVATE ${COMPONENT_NAME}
                                                       glog::glog)
  add_dependencies(xmodel_image_py_module py_dpu_model_param_proto)
endif(BUILD_PYTHON)
