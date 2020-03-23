#
#  Copyright 2019 Xilinx Inc.
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# 
macro(XILINX_AI_SDK_ADD_ACCURACY MODEL MODEL_NAME)
  #set(XILINX_AI_SDK_MODEL_NAME ${MODEL_NAME})
  if(${ARGC} GREATER 2 AND NOT ${ARGV2} STREQUAL "NONE")
	  set(XILINX_AI_SDK_NEED_CONFIG TRUE)
  else()
	  set(XILINX_AI_SDK_NEED_CONFIG FALSE)
  endif()
  #check_include_file_cxx(vitis/ai/${MODEL}.hpp XILINX_AI_SDK_HAVE_${MODEL}_HEADER)
  set(XILINX_AI_SDK_HAVE_${MODEL}_HEADER TRUE)
  if(XILINX_AI_SDK_HAVE_${MODEL}_HEADER)
	  #check_include_file_cxx(${CMAKE_SOURCE_DIR}/samples/${MODEL}/test_accuracy_${MODEL_NAME}.cpp.in XILINX_AI_SDK_NEED_CONFIG)
    if(XILINX_AI_SDK_NEED_CONFIG)
      configure_file(${CMAKE_SOURCE_DIR}/samples/${MODEL}/test_accuracy_${MODEL}.cpp.in ${CMAKE_BINARY_DIR}/samples/${MODEL}/test_accuracy_${MODEL_NAME}.cpp)
      set(CPP_PATH ${CMAKE_BINARY_DIR})
    else(XILINX_AI_SDK_NEED_CONFIG)
      set(CPP_PATH ${CMAKE_CURRENT_SOURCE_DIR})
    endif(XILINX_AI_SDK_NEED_CONFIG)
    add_executable(test_accuracy_${MODEL_NAME}  ${CPP_PATH}/samples/${MODEL}/test_accuracy_${MODEL_NAME}.cpp)
    target_link_libraries(test_accuracy_${MODEL_NAME} ${OpenCV_LIBS} ${PROJECT_NAME}::model_config ${PROJECT_NAME}::${MODEL} glog json-c xnnpp ${PROJECT_NAME}::math ${Pthread_LIB})
    install(TARGETS test_accuracy_${MODEL_NAME} DESTINATION ${SAMPLE_INATLL_PATH}/${MODEL})
    install(FILES ${CPP_PATH}/samples/${MODEL}/test_accuracy_${MODEL_NAME}.cpp DESTINATION ${SAMPLE_INATLL_PATH}/${MODEL})
  endif(XILINX_AI_SDK_HAVE_${MODEL}_HEADER)
endmacro()

macro(XILINX_AI_SDK_ADD_CUSTOMER_PROVIDED_MODEL_TEST TEST_CLASS)
    add_executable(test_customer_provided_model_${TEST_CLASS}  ${CMAKE_SOURCE_DIR}/samples/${TEST_CLASS}/test_customer_provided_model_${TEST_CLASS}.cpp)
    target_link_libraries(test_customer_provided_model_${TEST_CLASS} ${OpenCV_LIBS} dp${TEST_CLASS} glog ${PROJECT_NAME}::dpu_task ${PROJECT_NAME}::math ${Pthread_LIB})
    install(TARGETS test_customer_provided_model_${TEST_CLASS} DESTINATION ${SAMPLE_INATLL_PATH}/${TEST_CLASS})
    install(FILES ${CMAKE_SOURCE_DIR}/samples/${TEST_CLASS}/test_customer_provided_model_${TEST_CLASS}.cpp  DESTINATION ${SAMPLE_INATLL_PATH}/${TEST_CLASS})
endmacro()


macro(XILINX_AI_SDK_TEST_BY_NAME LIB_NAME MODEL_CLASS)
  set(XILINX_AI_SDK_TEMPLATE_FILE_SUFFIX "by_name")
  set(XILINX_AI_SDK_MODEL_NAME ${LIB_NAME})
  set(XILINX_AI_SDK_MODEL_WITH_VIDEO TRUE)
  if(${ARGC} GREATER 2)
  set(TMPSUFFIX "${ARGV2}")
    if (${TMPSUFFIX} STREQUAL "NO_VIDEO")
      set(XILINX_AI_SDK_MODEL_WITH_VIDEO FALSE)
      set(TMPSUFFIX "")
    endif()
  else(${ARGC} GREATER 2)
    set(TMPSUFFIX "")
  endif(${ARGC} GREATER 2)
  set(XILINX_AI_SDK_MODEL_CLASS ${MODEL_CLASS})
  #check_include_file_cxx(vitis/ai/${LIB_NAME}.hpp XILINX_AI_SDK_HAVE_${LIB_NAME}_HEADER)
  set(XILINX_AI_SDK_HAVE_${LIB_NAME}_HEADER TRUE)
  if(XILINX_AI_SDK_HAVE_${LIB_NAME}_HEADER)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/test_jpeg_xxx_${XILINX_AI_SDK_TEMPLATE_FILE_SUFFIX}.cpp.in ${CMAKE_BINARY_DIR}/samples/${LIB_NAME}/test_jpeg_${LIB_NAME}.cpp)
    add_executable(test_jpeg_${LIB_NAME} 
	    ${CMAKE_CURRENT_SOURCE_DIR}/samples/${LIB_NAME}/process_result.hpp
      ${CMAKE_BINARY_DIR}/samples/${LIB_NAME}/test_jpeg_${LIB_NAME}.cpp)
    #add_executable(test_jpeg_${LIB_NAME}  ${CMAKE_SOURCE_DIR}/samples/${LIB_NAME}/test_jpeg_${LIB_NAME}.cpp)
    target_include_directories(test_jpeg_${LIB_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/samples/${LIB_NAME}/)
    target_link_libraries(test_jpeg_${LIB_NAME} ${OpenCV_LIBS} ${PROJECT_NAME}::${LIB_NAME} glog  ${PROJECT_NAME}::xnnpp ${PROJECT_NAME}::math ${Pthread_LIB})
    install(TARGETS test_jpeg_${LIB_NAME} DESTINATION ${SAMPLE_INATLL_PATH}/${LIB_NAME})
    install(FILES ${CMAKE_BINARY_DIR}/samples/${LIB_NAME}/test_jpeg_${LIB_NAME}.cpp DESTINATION ${SAMPLE_INATLL_PATH}/${LIB_NAME})
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/samples/${LIB_NAME}/process_result.hpp DESTINATION ${SAMPLE_INATLL_PATH}/${LIB_NAME})

    if(XILINX_AI_SDK_MODEL_WITH_VIDEO)  
      configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/test_video_xxx_${XILINX_AI_SDK_TEMPLATE_FILE_SUFFIX}.cpp.in  ${CMAKE_BINARY_DIR}/samples/${LIB_NAME}/test_video_${LIB_NAME}${TMPSUFFIX}.cpp)
      add_executable(test_video_${LIB_NAME}${TMPSUFFIX}
        ${CMAKE_CURRENT_SOURCE_DIR}/samples/${LIB_NAME}/process_result.hpp
        ${CMAKE_BINARY_DIR}/samples/${LIB_NAME}/test_video_${LIB_NAME}${TMPSUFFIX}.cpp)
target_include_directories(test_video_${LIB_NAME}${TMPSUFFIX} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/samples/${LIB_NAME}/)
      target_link_libraries(test_video_${LIB_NAME}${TMPSUFFIX} ${OpenCV_LIBS} ${PROJECT_NAME}::${LIB_NAME} ${PROJECT_NAME}::xnnpp glog ${Pthread_LIB})
      if(HAVE_DRM)
        target_link_libraries(test_video_${LIB_NAME}${TMPSUFFIX} drm)
      endif(HAVE_DRM)
      install(TARGETS test_video_${LIB_NAME}${TMPSUFFIX} DESTINATION ${SAMPLE_INATLL_PATH}/${LIB_NAME})
      install(FILES ${CMAKE_BINARY_DIR}/samples/${LIB_NAME}/test_video_${LIB_NAME}${TMPSUFFIX}.cpp DESTINATION ${SAMPLE_INATLL_PATH}/${LIB_NAME})
    endif(XILINX_AI_SDK_MODEL_WITH_VIDEO)

    configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/test_performance_xxx_${XILINX_AI_SDK_TEMPLATE_FILE_SUFFIX}.cpp.in ${CMAKE_BINARY_DIR}/samples/${LIB_NAME}/test_performance_${LIB_NAME}${TMPSUFFIX}.cpp)
    add_executable(test_performance_${LIB_NAME}${TMPSUFFIX} ${CMAKE_BINARY_DIR}/samples/${LIB_NAME}/test_performance_${LIB_NAME}${TMPSUFFIX}.cpp)
    target_link_libraries(test_performance_${LIB_NAME}${TMPSUFFIX} ${OpenCV_LIBS} ${PROJECT_NAME}::${LIB_NAME} ${PROJECT_NAME}::xnnpp vart::util glog ${Pthread_LIB})
    install(TARGETS test_performance_${LIB_NAME}${TMPSUFFIX} DESTINATION ${SAMPLE_INATLL_PATH}/${LIB_NAME})
    install(FILES ${CMAKE_BINARY_DIR}/samples/${LIB_NAME}/test_performance_${LIB_NAME}${TMPSUFFIX}.cpp DESTINATION ${SAMPLE_INATLL_PATH}/${LIB_NAME})


    file(GLOB IMGS ${CMAKE_CURRENT_SOURCE_DIR}/samples/${LIB_NAME}/*.jpg ${CMAKE_CURRENT_SOURCE_DIR}/samples/${LIB_NAME}/build.sh ${CMAKE_CURRENT_SOURCE_DIR}/samples/${LIB_NAME}/readme ${CMAKE_CURRENT_SOURCE_DIR}/samples/${LIB_NAME}/*.list)
    foreach(IMG ${IMGS})
      install(FILES ${IMG} DESTINATION ${SAMPLE_INATLL_PATH}/${LIB_NAME})
    endforeach()
    #install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/samples/${LIB_NAME}/images DESTINATION ${SAMPLE_INATLL_PATH}/${LIB_NAME})

    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/samples/${LIB_NAME}/process_result.hpp DESTINATION ${SAMPLE_INATLL_PATH}/${LIB_NAME})

  else(XILINX_AI_SDK_HAVE_${LIB_NAME}_HEADER)
    message(FATAL_ERROR "no testing source codes are generated for model ${LIB_NAME}, because vitis/ai/${LIB_NAME}.hpp is not found")

  endif(XILINX_AI_SDK_HAVE_${LIB_NAME}_HEADER)
endmacro()
