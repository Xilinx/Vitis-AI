# Copyright 2022-2023 Advanced Micro Devices Inc.
#
# Distributed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
option(BUILD_DPU_MODEL "invoke dnnc to build dpu models or not." off)

add_definitions(-DDEEPHI_DPU)
# set(DPU_TYPE "4096FA" CACHE "STRING" "Type of dpu, supported list: [ 1024FA, 1152FA, 4096FA ].")

if(BUILD_DPU_MODEL)
  if ("${DPU_TYPE}" STREQUAL "")
    message(FATAL_ERROR "DPU_TYPE is not defined, it must be one of [ 800FA, 1024FA, 1152FA, 1600FA, 2304FA, 4096FA ]")
  else()
    string(TOUPPER "${DPU_TYPE}"  DPU_TYPE)
  endif()
endif(BUILD_DPU_MODEL)
if(BUILD_DPU_MODEL)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*|ARM.*")
    set(DPU_ARCH_TYPE "arm32")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64.*")
    set(DPU_ARCH_TYPE "arm64")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "cortexa9hf*")
    set(DPU_ARCH_TYPE "arm32")
  else()
    message(FATAL_ERROR "not a valid arch type ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
endif(BUILD_DPU_MODEL)

set(DNNC_WRAPPER ${CMAKE_CURRENT_LIST_DIR}/dnnc-wrapper)
function (ADD_DPU_MODEL MODEL_NAME)
  if(BUILD_DPU_MODEL)
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/libdpumodel${MODEL_NAME}.so
      COMMAND ${DNNC_WRAPPER}
      ARGS
      ${CMAKE_SOURCE_DIR}
      CAFFE
      ${CMAKE_BINARY_DIR}
      ${MODEL_NAME}
      ${DPU_TYPE}
      ${DPU_ARCH_TYPE}
      ${CMAKE_CXX_COMPILER}
      false
      DEPENDS ${CMAKE_SOURCE_DIR}/model/caffe/${MODEL_NAME}/deploy.prototxt ${CMAKE_SOURCE_DIR}/model/caffe/${MODEL_NAME}/deploy.caffemodel
      )
    install(FILES ${CMAKE_BINARY_DIR}/libdpumodel${MODEL_NAME}.so
      DESTINATION lib)
    add_custom_target(${MODEL_NAME} ALL
      DEPENDS ${CMAKE_BINARY_DIR}/libdpumodel${MODEL_NAME}.so)
  endif(BUILD_DPU_MODEL)
endfunction()

function (ADD_DPU_MODEL_TF MODEL_NAME)
  if(BUILD_DPU_MODEL)
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/libdpumodel${MODEL_NAME}.so
      COMMAND ${DNNC_WRAPPER}
      ARGS
      ${CMAKE_SOURCE_DIR}
      TENSORFLOW
      ${CMAKE_BINARY_DIR}
      ${MODEL_NAME}
      ${DPU_TYPE}
      ${DPU_ARCH_TYPE}
      ${CMAKE_CXX_COMPILER}
      false
      DEPENDS ${CMAKE_SOURCE_DIR}/model/tensorflow/${MODEL_NAME}/deploy.pb
      )
    install(FILES ${CMAKE_BINARY_DIR}/libdpumodel${MODEL_NAME}.so
      DESTINATION lib)
    add_custom_target(${MODEL_NAME} ALL
      DEPENDS ${CMAKE_BINARY_DIR}/libdpumodel${MODEL_NAME}.so)
  endif(BUILD_DPU_MODEL)
endfunction()

function (ADD_DPU_MODEL_WITH_FIX_INFO MODEL_NAME)
  if(BUILD_DPU_MODEL)
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/libdpumodel${MODEL_NAME}.so
      COMMAND ${DNNC_WRAPPER}
      ARGS
      ${CMAKE_SOURCE_DIR}
      CAFFE
      ${CMAKE_BINARY_DIR}
      ${MODEL_NAME}
      ${DPU_TYPE}
      ${DPU_ARCH_TYPE}
      ${CMAKE_CXX_COMPILER}
      true
      )
    install(FILES ${CMAKE_BINARY_DIR}/libdpumodel${MODEL_NAME}.so
      DESTINATION lib)
    add_custom_target(${MODEL_NAME} ALL
      DEPENDS ${CMAKE_BINARY_DIR}/libdpumodel${MODEL_NAME}.so)
  endif(BUILD_DPU_MODEL)
endfunction()

set(DNNC_ZOO_WRAPPER ${CMAKE_CURRENT_LIST_DIR}/dnnc-zoo-wrapper)
function (ADD_MODEL_FROM_ZOO MODEL_NAME URL_PATH)
  set(MODEL_TYPE caffe)
  if(BUILD_DPU_MODEL)
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/libdpumodel${MODEL_NAME}.so
      COMMAND ${DNNC_ZOO_WRAPPER}
      ARGS
      ${CMAKE_BINARY_DIR}
      ${MODEL_TYPE}
      ${MODEL_NAME}
      ${DPU_TYPE}
      ${DPU_ARCH_TYPE}
      ${CMAKE_CXX_COMPILER}
      ${URL_PATH}
      )
    install(FILES ${CMAKE_BINARY_DIR}/libdpumodel${MODEL_NAME}.so
      DESTINATION lib)
    add_custom_target(${MODEL_NAME} ALL
      DEPENDS ${CMAKE_BINARY_DIR}/libdpumodel${MODEL_NAME}.so)
  endif(BUILD_DPU_MODEL)
endfunction()

function (ADD_MODEL_FROM_ZOO_TF MODEL_NAME URL_PATH)
  set(MODEL_TYPE tensorflow)
  if(BUILD_DPU_MODEL)
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/libdpumodel${MODEL_NAME}.so
      COMMAND ${DNNC_ZOO_WRAPPER}
      ARGS
      ${CMAKE_BINARY_DIR}
      ${MODEL_TYPE}
      ${MODEL_NAME}
      ${DPU_TYPE}
      ${DPU_ARCH_TYPE}
      ${CMAKE_CXX_COMPILER}
      ${URL_PATH}
      )
    install(FILES ${CMAKE_BINARY_DIR}/libdpumodel${MODEL_NAME}.so
      DESTINATION lib)
    add_custom_target(${MODEL_NAME} ALL
      DEPENDS ${CMAKE_BINARY_DIR}/libdpumodel${MODEL_NAME}.so)
  endif(BUILD_DPU_MODEL)
endfunction()
