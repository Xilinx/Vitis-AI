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

# vai_overview_add_test
function(vai_overview_add_test name model model_class)
  set(options NO_VIDEO NO_CONFIG)
  set(oneValueArgs NAME)
  set(multiValueArgs REQUIRE)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  set(XILINX_AI_SDK_MODEL_NAME ${model})
  set(XILINX_AI_SDK_MODEL_CLASS ${model_class})

  set(XILINX_AI_SDK_BUILD_SCRIPT_REQUIRE -l${PROJECT_NAME}-${model})
  if(ARG_REQUIRE)
    foreach(_mod ${ARG_REQUIRE})
      if(TARGET ${_mod})
        get_target_property(_property ${_mod} LIBRARY_OUTPUT_NAME)
        list(APPEND _links -l${_property})
      else()
        list(APPEND _links -l${_mod})
      endif()
    endforeach(_mod)
    unset(XILINX_AI_SDK_BUILD_SCRIPT_REQUIRE)
    list(JOIN _links " " XILINX_AI_SDK_BUILD_SCRIPT_REQUIRE)
  endif(ARG_REQUIRE)

  # add build.sh
  configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/build.sh.in
                 ${CMAKE_BINARY_DIR}/samples/${model}/build.sh @ONLY)
  install(
    FILES ${CMAKE_BINARY_DIR}/samples/${model}/build.sh
    PERMISSIONS
      OWNER_READ
      OWNER_WRITE
      OWNER_EXECUTE
      GROUP_READ
      GROUP_WRITE
      GROUP_EXECUTE
      WORLD_READ
      WORLD_EXECUTE
    DESTINATION ${SAMPLE_INATLL_PATH}/${model})

  # add test_jpeg_xxx
  set(VAI_OVERVIEW_TEST_NEED_CONFIG TRUE)
  if(ARG_NO_CONFIG)
    set(VAI_OVERVIEW_TEST_NEED_CONFIG FALSE)
  endif(ARG_NO_CONFIG)

  set(EXE_NAME test_jpeg_${name})
  if(ARG_NAME)
    set(EXE_NAME ${ARG_NAME})
  endif(ARG_NAME)

  if(VAI_OVERVIEW_TEST_NEED_CONFIG)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/test_jpeg_xxx_by_name.cpp.in
                   ${CMAKE_BINARY_DIR}/samples/${model}/${EXE_NAME}.cpp)
    set(CPP_PATH ${CMAKE_BINARY_DIR})
  else(VAI_OVERVIEW_TEST_NEED_CONFIG)
    set(CPP_PATH ${CMAKE_CURRENT_SOURCE_DIR})
  endif(VAI_OVERVIEW_TEST_NEED_CONFIG)

  add_executable(${EXE_NAME} ${CPP_PATH}/samples/${model}/${EXE_NAME}.cpp)
  target_include_directories(
    ${EXE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/samples/${model}/)
  if(ARG_REQUIRE)
    target_link_libraries(${EXE_NAME} PRIVATE ${ARG_REQUIRE}
                                              ${PROJECT_NAME}::benchmark)
  else(ARG_REQUIRE)
    target_link_libraries(${EXE_NAME} PRIVATE ${PROJECT_NAME}::${model}
                                              ${PROJECT_NAME}::benchmark)
  endif(ARG_REQUIRE)

  install(TARGETS ${EXE_NAME} DESTINATION ${SAMPLE_INATLL_PATH}/${model})
  install(FILES ${CPP_PATH}/samples/${model}/${EXE_NAME}.cpp
          DESTINATION ${SAMPLE_INATLL_PATH}/${model})
  file(GLOB _files ${CMAKE_CURRENT_SOURCE_DIR}/samples/${model}/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/samples/${model}/readme)
  foreach(_file ${_files})
    install(FILES ${_file} DESTINATION ${SAMPLE_INATLL_PATH}/${model})
  endforeach(_file)

  # add test_video_xxx
  set(VAI_OVERVIEW_TEST_WITH_VIDEO TRUE)
  if(ARG_NO_VIDEO)
    set(VAI_OVERVIEW_TEST_WITH_VIDEO FALSE)
  endif(ARG_NO_VIDEO)
  if(VAI_OVERVIEW_TEST_WITH_VIDEO)
    if(VAI_OVERVIEW_TEST_NEED_CONFIG)
      configure_file(
        ${CMAKE_CURRENT_LIST_DIR}/cmake/test_video_xxx_by_name.cpp.in
        ${CMAKE_BINARY_DIR}/samples/${model}/test_video_${name}.cpp)
      set(CPP_PATH ${CMAKE_BINARY_DIR})
    else(VAI_OVERVIEW_TEST_NEED_CONFIG)
      set(CPP_PATH ${CMAKE_CURRENT_SOURCE_DIR})
    endif(VAI_OVERVIEW_TEST_NEED_CONFIG)
    add_executable(test_video_${name}
                   ${CPP_PATH}/samples/${model}/test_video_${name}.cpp)
    target_include_directories(
      test_video_${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/samples/${name}/)
    if(ARG_REQUIRE)
      target_link_libraries(test_video_${name}
                            PRIVATE ${ARG_REQUIRE} ${PROJECT_NAME}::benchmark)
    else(ARG_REQUIRE)
      target_link_libraries(
        test_video_${name} PRIVATE ${PROJECT_NAME}::${model}
                                   ${PROJECT_NAME}::benchmark)
    endif(ARG_REQUIRE)
    if(HAVE_DRM)
      target_link_libraries(test_video_${name} drm)
    endif(HAVE_DRM)
    install(TARGETS test_video_${name}
            DESTINATION ${SAMPLE_INATLL_PATH}/${model})
    install(FILES ${CPP_PATH}/samples/${model}/test_video_${name}.cpp
            DESTINATION ${SAMPLE_INATLL_PATH}/${model})
  endif(VAI_OVERVIEW_TEST_WITH_VIDEO)
endfunction(vai_overview_add_test)

# vai_overview_add_performance
function(vai_overview_add_performance name model model_class)
  set(options NO_CONFIG)
  set(oneValueArgs "")
  set(multiValueArgs REQUIRE)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  set(XILINX_AI_SDK_MODEL_NAME ${model})
  set(XILINX_AI_SDK_MODEL_CLASS ${model_class})
  set(VAI_OVERVIEW_PERFORMANCE_NEED_CONFIG TRUE)
  if(ARG_NO_CONFIG)
    set(VAI_OVERVIEW_PERFORMANCE_NEED_CONFIG FALSE)
  endif(ARG_NO_CONFIG)
  if(VAI_OVERVIEW_PERFORMANCE_NEED_CONFIG)
    configure_file(
      ${CMAKE_CURRENT_LIST_DIR}/cmake/test_performance_xxx_by_name.cpp.in
      ${CMAKE_BINARY_DIR}/samples/${model}/test_performance_${name}.cpp)
    set(CPP_PATH ${CMAKE_BINARY_DIR})
  else(VAI_OVERVIEW_PERFORMANCE_NEED_CONFIG)
    set(CPP_PATH ${CMAKE_CURRENT_SOURCE_DIR})
  endif(VAI_OVERVIEW_PERFORMANCE_NEED_CONFIG)

  add_executable(test_performance_${name}
                 ${CPP_PATH}/samples/${model}/test_performance_${name}.cpp)
  target_include_directories(
    test_performance_${name}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/samples/${model}/)
  if(ARG_REQUIRE)
    target_link_libraries(test_performance_${name}
                          PRIVATE ${ARG_REQUIRE} ${PROJECT_NAME}::benchmark)
  else(ARG_REQUIRE)
    target_link_libraries(
      test_performance_${name} PRIVATE ${PROJECT_NAME}::${model}
                                       ${PROJECT_NAME}::benchmark)
  endif(ARG_REQUIRE)
  install(TARGETS test_performance_${name}
          DESTINATION ${SAMPLE_INATLL_PATH}/${model})
  install(FILES ${CPP_PATH}/samples/${model}/test_performance_${name}.cpp
          DESTINATION ${SAMPLE_INATLL_PATH}/${model})
endfunction(vai_overview_add_performance)

# vai_overview_add_accuracy
function(vai_overview_add_accuracy name model)
  set(options "")
  set(oneValueArgs ENABLE_IF)
  set(multiValueArgs REQUIRE)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  add_executable(
    test_accuracy_${name}
    ${CMAKE_CURRENT_SOURCE_DIR}/samples/${model}/test_accuracy_${name}.cpp)

  if(ARG_REQUIRE)
    target_link_libraries(test_accuracy_${name}
                          PRIVATE ${ARG_REQUIRE} ${PROJECT_NAME}::benchmark)
  else(ARG_REQUIRE)
    target_link_libraries(
      test_accuracy_${name} PRIVATE ${PROJECT_NAME}::${model}
                                    ${PROJECT_NAME}::benchmark)
  endif(ARG_REQUIRE)

  install(TARGETS test_accuracy_${name}
          DESTINATION ${SAMPLE_INATLL_PATH}/${model})
  install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/samples/${model}/test_accuracy_${name}.cpp
    DESTINATION ${SAMPLE_INATLL_PATH}/${model})
endfunction(vai_overview_add_accuracy)

# vai_overview_add_app
function(vai_overview_add_app)
  set(options "")
  set(oneValueArgs NAME)
  set(multiValueArgs SRCS REQUIRE VAI_INSTALL_FILES VAI_INSTALL_BUILD)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  add_executable(${ARG_NAME} ${ARG_SRCS})
  if(ARG_REQUIRE)
    target_link_libraries(${ARG_NAME} PRIVATE ${ARG_REQUIRE})
  endif(ARG_REQUIRE)
  if(HAVE_DRM)
    target_link_libraries(${ARG_NAME} drm)
  endif(HAVE_DRM)

  install(TARGETS ${ARG_NAME} DESTINATION ${DEMO_INATLL_PATH}/${ARG_NAME})
  if(ARG_VAI_INSTALL_FILES)
    foreach(_file ${ARG_VAI_INSTALL_FILES})
      install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/${_file}
              DESTINATION ${DEMO_INATLL_PATH}/${ARG_NAME})
    endforeach(_file)
  endif(ARG_VAI_INSTALL_FILES)
  if(ARG_VAI_INSTALL_BUILD)
    foreach(_file ${ARG_VAI_INSTALL_BUILD})
      install(
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/${_file}
        PERMISSIONS
          OWNER_READ
          OWNER_WRITE
          OWNER_EXECUTE
          GROUP_READ
          GROUP_WRITE
          GROUP_EXECUTE
          WORLD_READ
          WORLD_EXECUTE
        DESTINATION ${DEMO_INATLL_PATH}/${ARG_NAME})
    endforeach(_file)
  endif(ARG_VAI_INSTALL_BUILD)
endfunction(vai_overview_add_app)

# vai_overview_add_demo
function(vai_overview_add_demo)
  set(options "")
  set(oneValueArgs NAME)
  set(multiValueArgs SRCS REQUIRE VAI_INSTALL_FILES VAI_INSTALL_BUILD)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  add_executable(demo_${ARG_NAME} ${ARG_SRCS})
  if(ARG_REQUIRE)
    target_link_libraries(demo_${ARG_NAME} PRIVATE ${ARG_REQUIRE})
  endif(ARG_REQUIRE)

  install(TARGETS demo_${ARG_NAME}
          DESTINATION ${SAMPLE_INATLL_PATH}/dpu_task/${ARG_NAME})
  if(ARG_VAI_INSTALL_FILES)
    foreach(_file ${ARG_VAI_INSTALL_FILES})
      install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/${_file}
              DESTINATION ${SAMPLE_INATLL_PATH}/dpu_task/${ARG_NAME})
    endforeach(_file)
  endif(ARG_VAI_INSTALL_FILES)
  if(ARG_VAI_INSTALL_BUILD)
    foreach(_file ${ARG_VAI_INSTALL_BUILD})
      install(
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/${_file}
        PERMISSIONS
          OWNER_READ
          OWNER_WRITE
          OWNER_EXECUTE
          GROUP_READ
          GROUP_WRITE
          GROUP_EXECUTE
          WORLD_READ
          WORLD_EXECUTE
        DESTINATION ${SAMPLE_INATLL_PATH}/dpu_task/${ARG_NAME})
    endforeach(_file)
  endif(ARG_VAI_INSTALL_BUILD)

endfunction(vai_overview_add_demo)

# vai_overview_add_dpu_task
function(vai_overview_add_dpu_task)
  set(options "")
  set(oneValueArgs NAME)
  set(multiValueArgs SRCS REQUIRE VAI_INSTALL_FOLDER VAI_INSTALL_BUILD)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  file(GLOB_RECURSE ARG_SRCS_OBJ ${ARG_VAI_INSTALL_FOLDER}/*)
  add_executable(demo_${ARG_NAME} ${ARG_VAI_INSTALL_FOLDER}/demo_${ARG_NAME}.cpp
                                  ${ARG_SRCS})
  add_executable(
    test_performance_${ARG_NAME}
    ${ARG_VAI_INSTALL_FOLDER}/test_performance_${ARG_NAME}.cpp ${ARG_SRCS})
  if(ARG_REQUIRE)
    target_link_libraries(demo_${ARG_NAME} PRIVATE ${ARG_REQUIRE})
    target_link_libraries(test_performance_${ARG_NAME} PRIVATE ${ARG_REQUIRE})
  endif(ARG_REQUIRE)

  install(TARGETS demo_${ARG_NAME}
          DESTINATION ${SAMPLE_INATLL_PATH}/dpu_task/${ARG_NAME})
  install(TARGETS test_performance_${ARG_NAME}
          DESTINATION ${SAMPLE_INATLL_PATH}/dpu_task/${ARG_NAME})

  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_VAI_INSTALL_FOLDER}/demo_${ARG_NAME}_v2.cpp)
    add_executable(demo_${ARG_NAME}_v2 ${ARG_VAI_INSTALL_FOLDER}/demo_${ARG_NAME}_v2.cpp
      ${ARG_VAI_INSTALL_FOLDER}/${ARG_NAME}_v2.cpp)
    add_executable(
      test_performance_${ARG_NAME}_v2
      ${ARG_VAI_INSTALL_FOLDER}/test_performance_${ARG_NAME}_v2.cpp
      ${ARG_VAI_INSTALL_FOLDER}/${ARG_NAME}_v2.cpp)
    target_link_libraries(demo_${ARG_NAME}_v2 PRIVATE ${ARG_REQUIRE})
    target_link_libraries(test_performance_${ARG_NAME}_v2 PRIVATE ${ARG_REQUIRE})

    install(TARGETS demo_${ARG_NAME}_v2
          DESTINATION ${SAMPLE_INATLL_PATH}/dpu_task/${ARG_NAME})
    install(TARGETS test_performance_${ARG_NAME}_v2
          DESTINATION ${SAMPLE_INATLL_PATH}/dpu_task/${ARG_NAME})
  endif()
  
  if(ARG_VAI_INSTALL_FOLDER)
    foreach(_file ${ARG_SRCS_OBJ})
      install(FILES ${_file}
              DESTINATION ${SAMPLE_INATLL_PATH}/dpu_task/${ARG_NAME})
    endforeach(_file)
  endif(ARG_VAI_INSTALL_FOLDER)
  if(ARG_VAI_INSTALL_BUILD)
    foreach(_file ${ARG_VAI_INSTALL_BUILD})
      install(
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/${_file}
        PERMISSIONS
          OWNER_READ
          OWNER_WRITE
          OWNER_EXECUTE
          GROUP_READ
          GROUP_WRITE
          GROUP_EXECUTE
          WORLD_READ
          WORLD_EXECUTE
        DESTINATION ${SAMPLE_INATLL_PATH}/dpu_task/${ARG_NAME})
    endforeach(_file)
  endif(ARG_VAI_INSTALL_BUILD)
endfunction(vai_overview_add_dpu_task)

macro(XILINX_AI_SDK_ADD_ACCURACY MODEL MODEL_NAME)
  # set(XILINX_AI_SDK_MODEL_NAME ${MODEL_NAME})
  if(${ARGC} GREATER 2 AND NOT ${ARGV2} STREQUAL "NONE")
    set(XILINX_AI_SDK_NEED_CONFIG TRUE)
  else()
    set(XILINX_AI_SDK_NEED_CONFIG FALSE)
  endif()
  # check_include_file_cxx(vitis/ai/${MODEL}.hpp
  # XILINX_AI_SDK_HAVE_${MODEL}_HEADER)
  set(XILINX_AI_SDK_HAVE_${MODEL}_HEADER TRUE)
  if(XILINX_AI_SDK_HAVE_${MODEL}_HEADER)
    # check_include_file_cxx(${CMAKE_SOURCE_DIR}/samples/${MODEL}/test_accuracy_${MODEL_NAME}.cpp.in
    # XILINX_AI_SDK_NEED_CONFIG)
    if(XILINX_AI_SDK_NEED_CONFIG)
      configure_file(
        ${CMAKE_SOURCE_DIR}/samples/${MODEL}/test_accuracy_${MODEL}.cpp.in
        ${CMAKE_BINARY_DIR}/samples/${MODEL}/test_accuracy_${MODEL_NAME}.cpp)
      set(CPP_PATH ${CMAKE_BINARY_DIR})
    else(XILINX_AI_SDK_NEED_CONFIG)
      set(CPP_PATH ${CMAKE_CURRENT_SOURCE_DIR})
    endif(XILINX_AI_SDK_NEED_CONFIG)
    add_executable(test_accuracy_${MODEL_NAME}
                   ${CPP_PATH}/samples/${MODEL}/test_accuracy_${MODEL_NAME}.cpp)
    target_link_libraries(
      test_accuracy_${MODEL_NAME}
      ${OpenCV_LIBS}
      ${PROJECT_NAME}::model_config
      ${PROJECT_NAME}::${MODEL}
      glog::glog
      json-c
      ${PROJECT_NAME}::xnnpp
      ${PROJECT_NAME}::math
      ${Pthread_LIB})
    install(TARGETS test_accuracy_${MODEL_NAME}
            DESTINATION ${SAMPLE_INATLL_PATH}/${MODEL})
    install(FILES ${CPP_PATH}/samples/${MODEL}/test_accuracy_${MODEL_NAME}.cpp
            DESTINATION ${SAMPLE_INATLL_PATH}/${MODEL})
  endif(XILINX_AI_SDK_HAVE_${MODEL}_HEADER)
endmacro()

macro(XILINX_AI_SDK_ADD_CUSTOMER_PROVIDED_MODEL_TEST TEST_CLASS)
  add_executable(
    test_customer_provided_model_${TEST_CLASS}
    ${CMAKE_SOURCE_DIR}/samples/${TEST_CLASS}/test_customer_provided_model_${TEST_CLASS}.cpp
  )
  target_link_libraries(
    test_customer_provided_model_${TEST_CLASS}
    ${OpenCV_LIBS}
    dp${TEST_CLASS}
    glog::glog
    ${PROJECT_NAME}::dpu_task
    ${PROJECT_NAME}::math
    ${Pthread_LIB})
  install(TARGETS test_customer_provided_model_${TEST_CLASS}
          DESTINATION ${SAMPLE_INATLL_PATH}/${TEST_CLASS})
  install(
    FILES
      ${CMAKE_SOURCE_DIR}/samples/${TEST_CLASS}/test_customer_provided_model_${TEST_CLASS}.cpp
    DESTINATION ${SAMPLE_INATLL_PATH}/${TEST_CLASS})
endmacro()

macro(XILINX_AI_SDK_TEST_BY_NAME LIB_NAME MODEL_CLASS)
  set(XILINX_AI_SDK_TEMPLATE_FILE_SUFFIX "by_name")
  set(XILINX_AI_SDK_MODEL_NAME ${LIB_NAME})
  set(XILINX_AI_SDK_MODEL_WITH_VIDEO TRUE)
  if(${ARGC} GREATER 2)
    set(TMPSUFFIX "${ARGV2}")
    if(${TMPSUFFIX} STREQUAL "NO_VIDEO")
      set(XILINX_AI_SDK_MODEL_WITH_VIDEO FALSE)
      set(TMPSUFFIX "")
    endif()
  else(${ARGC} GREATER 2)
    set(TMPSUFFIX "")
  endif(${ARGC} GREATER 2)
  set(XILINX_AI_SDK_MODEL_CLASS ${MODEL_CLASS})
  # check_include_file_cxx(vitis/ai/${LIB_NAME}.hpp
  # XILINX_AI_SDK_HAVE_${LIB_NAME}_HEADER)
  set(XILINX_AI_SDK_HAVE_${LIB_NAME}_HEADER TRUE)
  if(XILINX_AI_SDK_HAVE_${LIB_NAME}_HEADER)
    configure_file(
      ${CMAKE_CURRENT_LIST_DIR}/cmake/test_jpeg_xxx_${XILINX_AI_SDK_TEMPLATE_FILE_SUFFIX}.cpp.in
      ${CMAKE_BINARY_DIR}/samples/${LIB_NAME}/test_jpeg_${LIB_NAME}.cpp)
    add_executable(
      test_jpeg_${LIB_NAME}
      ${CMAKE_CURRENT_SOURCE_DIR}/samples/${LIB_NAME}/process_result.hpp
      ${CMAKE_BINARY_DIR}/samples/${LIB_NAME}/test_jpeg_${LIB_NAME}.cpp)
    # add_executable(test_jpeg_${LIB_NAME}
    # ${CMAKE_SOURCE_DIR}/samples/${LIB_NAME}/test_jpeg_${LIB_NAME}.cpp)
    target_include_directories(
      test_jpeg_${LIB_NAME}
      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/samples/${LIB_NAME}/)
    target_link_libraries(
      test_jpeg_${LIB_NAME}
      ${OpenCV_LIBS}
      ${PROJECT_NAME}::${LIB_NAME}
      glog::glog
      ${PROJECT_NAME}::xnnpp
      ${PROJECT_NAME}::math
      ${Pthread_LIB}
      ${PROJECT_NAME}::benchmark
      ${PROJECT_NAME}::dpu_task)
    install(TARGETS test_jpeg_${LIB_NAME}
            DESTINATION ${SAMPLE_INATLL_PATH}/${LIB_NAME})
    install(
      FILES ${CMAKE_BINARY_DIR}/samples/${LIB_NAME}/test_jpeg_${LIB_NAME}.cpp
      DESTINATION ${SAMPLE_INATLL_PATH}/${LIB_NAME})
    install(
      FILES ${CMAKE_CURRENT_SOURCE_DIR}/samples/${LIB_NAME}/process_result.hpp
      DESTINATION ${SAMPLE_INATLL_PATH}/${LIB_NAME})

    if(XILINX_AI_SDK_MODEL_WITH_VIDEO)
      configure_file(
        ${CMAKE_CURRENT_LIST_DIR}/cmake/test_video_xxx_${XILINX_AI_SDK_TEMPLATE_FILE_SUFFIX}.cpp.in
        ${CMAKE_BINARY_DIR}/samples/${LIB_NAME}/test_video_${LIB_NAME}${TMPSUFFIX}.cpp
      )
      add_executable(
        test_video_${LIB_NAME}${TMPSUFFIX}
        ${CMAKE_CURRENT_SOURCE_DIR}/samples/${LIB_NAME}/process_result.hpp
        ${CMAKE_BINARY_DIR}/samples/${LIB_NAME}/test_video_${LIB_NAME}${TMPSUFFIX}.cpp
      )
      target_include_directories(
        test_video_${LIB_NAME}${TMPSUFFIX}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/samples/${LIB_NAME}/)
      target_link_libraries(
        test_video_${LIB_NAME}${TMPSUFFIX}
        ${OpenCV_LIBS}
        ${PROJECT_NAME}::${LIB_NAME}
        ${PROJECT_NAME}::xnnpp
        glog::glog
        ${Pthread_LIB}
        ${PROJECT_NAME}::benchmark
        ${PROJECT_NAME}::dpu_task)
      if(HAVE_DRM)
        target_link_libraries(test_video_${LIB_NAME}${TMPSUFFIX} drm)
      endif(HAVE_DRM)
      install(TARGETS test_video_${LIB_NAME}${TMPSUFFIX}
              DESTINATION ${SAMPLE_INATLL_PATH}/${LIB_NAME})
      install(
        FILES
          ${CMAKE_BINARY_DIR}/samples/${LIB_NAME}/test_video_${LIB_NAME}${TMPSUFFIX}.cpp
        DESTINATION ${SAMPLE_INATLL_PATH}/${LIB_NAME})
    endif(XILINX_AI_SDK_MODEL_WITH_VIDEO)

    configure_file(
      ${CMAKE_CURRENT_LIST_DIR}/cmake/test_performance_xxx_${XILINX_AI_SDK_TEMPLATE_FILE_SUFFIX}.cpp.in
      ${CMAKE_BINARY_DIR}/samples/${LIB_NAME}/test_performance_${LIB_NAME}${TMPSUFFIX}.cpp
    )
    add_executable(
      test_performance_${LIB_NAME}${TMPSUFFIX}
      ${CMAKE_BINARY_DIR}/samples/${LIB_NAME}/test_performance_${LIB_NAME}${TMPSUFFIX}.cpp
    )
    target_link_libraries(
      test_performance_${LIB_NAME}${TMPSUFFIX}
      ${OpenCV_LIBS}
      ${PROJECT_NAME}::${LIB_NAME}
      ${PROJECT_NAME}::xnnpp
      vart::util
      glog::glog
      ${Pthread_LIB}
      ${PROJECT_NAME}::benchmark
      ${PROJECT_NAME}::dpu_task)
    install(TARGETS test_performance_${LIB_NAME}${TMPSUFFIX}
            DESTINATION ${SAMPLE_INATLL_PATH}/${LIB_NAME})
    install(
      FILES
        ${CMAKE_BINARY_DIR}/samples/${LIB_NAME}/test_performance_${LIB_NAME}${TMPSUFFIX}.cpp
      DESTINATION ${SAMPLE_INATLL_PATH}/${LIB_NAME})

    file(GLOB IMGS ${CMAKE_CURRENT_SOURCE_DIR}/samples/${LIB_NAME}/*.jpg
         ${CMAKE_CURRENT_SOURCE_DIR}/samples/${LIB_NAME}/readme
         ${CMAKE_CURRENT_SOURCE_DIR}/samples/${LIB_NAME}/*.list)
    foreach(IMG ${IMGS})
      install(FILES ${IMG} DESTINATION ${SAMPLE_INATLL_PATH}/${LIB_NAME})
    endforeach()

    install(
      FILES ${CMAKE_CURRENT_SOURCE_DIR}/samples/${LIB_NAME}/build.sh
      PERMISSIONS
        OWNER_READ
        OWNER_WRITE
        OWNER_EXECUTE
        GROUP_READ
        GROUP_EXECUTE
        WORLD_READ
        WORLD_EXECUTE
      DESTINATION ${SAMPLE_INATLL_PATH}/${LIB_NAME})

    # install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/samples/${LIB_NAME}/images
    # DESTINATION ${SAMPLE_INATLL_PATH}/${LIB_NAME})

    install(
      FILES ${CMAKE_CURRENT_SOURCE_DIR}/samples/${LIB_NAME}/process_result.hpp
      DESTINATION ${SAMPLE_INATLL_PATH}/${LIB_NAME})

  else(XILINX_AI_SDK_HAVE_${LIB_NAME}_HEADER)
    message(
      FATAL_ERROR
        "no testing source codes are generated for model ${LIB_NAME}, because vitis/ai/${LIB_NAME}.hpp is not found"
    )

  endif(XILINX_AI_SDK_HAVE_${LIB_NAME}_HEADER)
endmacro()

#for onnx
function(vai_overview_add_onnx name model)
  set(options NO_ACC)
  set(oneValueArgs ENABLE_IF)
  set(multiValueArgs REQUIRE)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${model}
    DESTINATION ${SAMPLE_ONNX_INSTALL_PATH})

  add_executable( test_${name}_onnx
    ${CMAKE_CURRENT_SOURCE_DIR}/${model}/test_${name}_onnx.cpp)

  if(ARG_REQUIRE)
    target_link_libraries(test_${name}_onnx PRIVATE ${ARG_REQUIRE} ${ORT_LIBRARY}  XRT::XRT vart-xrt-device-handle  glog::glog ${OpenCV_LIBS} ${PROJECT_NAME}::benchmark vart-util)
  else(ARG_REQUIRE)
    target_link_libraries(test_${name}_onnx PRIVATE ${ORT_LIBRARY}  XRT::XRT vart-xrt-device-handle glog::glog ${OpenCV_LIBS} ${PROJECT_NAME}::benchmark vart-util)
  endif(ARG_REQUIRE)
  install(TARGETS test_${name}_onnx DESTINATION ${SAMPLE_ONNX_INSTALL_PATH}/${model})

  add_executable( test_performance_${name}_onnx
    ${CMAKE_CURRENT_SOURCE_DIR}/${model}/test_performance_${name}_onnx.cpp)
  if(ARG_REQUIRE)
    target_link_libraries(test_performance_${name}_onnx PRIVATE ${ORT_LIBRARY} XRT::XRT vart-xrt-device-handle  glog::glog ${ARG_REQUIRE} ${OpenCV_LIBS} ${PROJECT_NAME}::benchmark vart-util)
  else(ARG_REQUIRE)
    target_link_libraries( test_performance_${name}_onnx PRIVATE ${ORT_LIBRARY}  XRT::XRT vart-xrt-device-handle glog::glog ${OpenCV_LIBS} ${PROJECT_NAME}::benchmark vart-util)
  endif(ARG_REQUIRE)
  install(TARGETS test_performance_${name}_onnx DESTINATION ${SAMPLE_ONNX_INSTALL_PATH}/${model})
  
  set(VAI_OVERVIEW_ONNX_WITH_ACC TRUE)
  if(ARG_NO_ACC)
    set(VAI_OVERVIEW_ONNX_WITH_ACC FALSE)
  endif(ARG_NO_ACC)

  if (VAI_OVERVIEW_ONNX_WITH_ACC) 
    add_executable( test_accuracy_${name}_onnx
      ${CMAKE_CURRENT_SOURCE_DIR}/${model}/test_accuracy_${name}_onnx.cpp)
    if(ARG_REQUIRE)
      target_link_libraries(test_accuracy_${name}_onnx PRIVATE ${ORT_LIBRARY}  XRT::XRT vart-xrt-device-handle  glog::glog ${ARG_REQUIRE} ${OpenCV_LIBS} ${PROJECT_NAME}::benchmark vart-util)
    else(ARG_REQUIRE)
      target_link_libraries( test_accuracy_${name}_onnx PRIVATE ${ORT_LIBRARY}   XRT::XRT vart-xrt-device-handle glog::glog ${OpenCV_LIBS} ${PROJECT_NAME}::benchmark vart-util)
    endif(ARG_REQUIRE)
    install(TARGETS test_accuracy_${name}_onnx DESTINATION ${SAMPLE_ONNX_INSTALL_PATH}/${model})
  endif(VAI_OVERVIEW_ONNX_WITH_ACC) 

endfunction(vai_overview_add_onnx)

