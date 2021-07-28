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
  SRCS include/vart/op_imp.h src/tensor_buffer_proxy.hpp
       src/tensor_buffer_proxy.cpp src/cpu_task.hpp src/cpu_task.cpp
       src/op_imp.cpp
  PUBLIC_REQUIRE ${PROJECT_NAME}::runner_helper
  PRIVATE_REQUIRE vart::runner vart::runner-assistant dl)

function(add_op_imp OP_TYPE)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs PUBLIC_REQUIRE PRIVATE_REQUIRE)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})
  add_library(${OP_TYPE} SHARED ${ARG_UNPARSED_ARGUMENTS})
  target_link_libraries(
    ${OP_TYPE}
    PUBLIC ${ARG_PUBLIC_REQUIRE}
    PRIVATE ${COMPONENT_NAME} ${ARG_PRIVATE_REQUIRE})
  set_target_properties(${OP_TYPE} PROPERTIES LIBRARY_OUTPUT_NAME
                                              vart_op_imp_${OP_TYPE})
  target_include_directories(
    ${OP_TYPE} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                      $<INSTALL_INTERFACE:include>)
  install(
    TARGETS ${OP_TYPE}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib)
endfunction(add_op_imp)

add_op_imp(softmax ops/softmax/softmax.cpp)
add_op_imp(fix2float ops/fix2float/fix2float.cpp)
add_op_imp(float2fix ops/float2fix/float2fix.cpp)
add_op_imp(fix ops/fix/fix.cpp)
add_op_imp(topk ops/topk/topk.cpp)
add_op_imp(avgpool2d ops/avgpool2d/avgpool2d.cpp)
add_op_imp(compare ops/compare/compare.cpp PRIVATE_REQUIRE crypto)
add_op_imp(const ops/const/const.cpp)
add_op_imp(const-fix ops/const-fix/const-fix.cpp)
add_op_imp(data-fix ops/data-fix/data-fix.cpp PRIVATE_REQUIRE crypto)
add_op_imp(depthwise-conv2d-fix
           ops/depthwise-conv2d-fix/depthwise-conv2d-fix.cpp)
add_op_imp(
  transposed-depthwise-conv2d-fix
  ops/transposed-depthwise-conv2d-fix/transposed-depthwise-conv2d-fix.cpp)
add_op_imp(upsample-fix ops/upsample-fix/upsample-fix.cpp)
add_op_imp(priorbox ops/priorbox/prior_box_layer.cpp
           ops/priorbox/prior_box_layer.hpp)
add_op_imp(reshape-fix ops/reshape-fix/reshape-fix.cpp)
add_op_imp(reshape ops/reshape-fix/reshape-fix.cpp)
add_op_imp(flatten ops/reshape-fix/reshape-fix.cpp)
add_op_imp(squeeze ops/reshape-fix/reshape-fix.cpp)
add_op_imp(reduction_max ops/reduce_max/reduce_max.cpp)
add_op_imp(sub ops/sub/sub.cpp)
add_op_imp(min ops/min/min.cpp)
add_op_imp(max ops/max/max.cpp)
add_op_imp(exp ops/exp/exp.cpp)
add_op_imp(reduction_sum ops/reduction_sum/reduction_sum.cpp)
add_op_imp(div ops/div/div.cpp)
add_op_imp(mul ops/mul/mul.cpp)
add_op_imp(concat-fix ops/concat/concat.cpp)
add_op_imp(concat ops/concat/concat.cpp)
add_op_imp(batchnorm ops/batchnorm/batchnorm.cpp)
add_op_imp(sigmoid ops/sigmoid/sigmoid.cpp)
add_op_imp(pool-fix ops/pool-fix/pool-fix.cpp)
add_op_imp(transpose ops/transpose/transpose.cpp PRIVATE_REQUIRE vart::util)
add_op_imp(strided_slice ops/strided_slice/strided_slice.cpp PRIVATE_REQUIRE
           vart::util)
add_op_imp(pad-fix ops/pad-fix/pad-fix.cpp)

if(BUILD_PYTHON)
  add_op_imp(python-cpu-op ops/python-cpu-op/python-cpu-op.cpp)
  if(CMAKE_CROSSCOMPILING)
    set_target_properties(python-cpu-op PROPERTIES INCLUDE_DIRECTORIES
                                                   ${VAI_PYTHON_INCLUDE_DIRS})
    target_link_libraries(python-cpu-op PRIVATE -l${VAI_PYTHON_LIB})
  else(CMAKE_CROSSCOMPILING)
    target_link_libraries(python-cpu-op PRIVATE ${PYTHON_LIBRARIES} vart::util)
    set_property(
      TARGET python-cpu-op
      APPEND
      PROPERTY INCLUDE_DIRECTORIES "${PYTHON_INCLUDE_DIRS}")
  endif(CMAKE_CROSSCOMPILING)
endif(BUILD_PYTHON)

vai_add_test(test_cpu_task)
vai_add_test(test_op_imp test/test_op_imp.cpp)
vai_add_test(test_experimental test/test_experimental.cpp)
vai_add_test(show_binary_image)
vai_add_test(check_supported_op test/check_supported_op.cpp)
target_link_libraries(check_supported_op dl)

add_test(
  NAME test_softmax
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test/test_softmax.py -v
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
