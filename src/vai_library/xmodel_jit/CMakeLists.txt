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

vai_add_library(SRCS ./include/vitis/ai/xmodel_jit.hpp src/xmodel_jit.cpp
  PRIVATE_REQUIRE dl xir::xir glog::glog)
if(BUILD_PYTHON)
  vai_add_library(
    NAME xmodel_jit_python
    SRCS src/xmodel_jit_python.hpp src/xmodel_jit_python.cpp
    PRIVATE_REQUIRE ${COMPONENT_NAME} extra_ops xir::xir)

  set_target_properties(xmodel_jit_python PROPERTIES INCLUDE_DIRECTORIES
    "${VAI_PYTHON_INCLUDE_DIRS};${PYTHON_INCLUDE_DIRS}")
  target_link_libraries(
    xmodel_jit_python
    PRIVATE
    ${PROJECT_NAME}::${COMPONENT_NAME}
    vart::util
    )

  if(VAI_PYTHON_LIB)
    target_link_libraries(
      xmodel_jit_python
      PRIVATE -l${VAI_PYTHON_LIB}
      )
  endif(VAI_PYTHON_LIB)

  if(PYTHON_LIBRARIES)
    target_link_libraries(
      xmodel_jit_python
      PRIVATE ${PYTHON_LIBRARIES}
      )
  endif(PYTHON_LIBRARIES)

  install(
    TARGETS xmodel_jit_python
    RUNTIME DESTINATION ${INSTALL_BIN_DIR}
    LIBRARY DESTINATION ${INSTALL_LIB_DIR})
endif(BUILD_PYTHON)
