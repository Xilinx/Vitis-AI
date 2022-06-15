# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

set(CMAKE_CXX_STANDARD 17)

# Source Files
set(LOG_SRC_LIST
    AksAIGraph.cpp
    AksAIGraphInAction.cpp
    AksAIGraphNode.cpp
    AksCommonUtils.cpp
    AksKernelDef.cpp
    AksLogger.cpp
    AksNodeParams.cpp
    AksParamProps.cpp
    AksSysManager.cpp
    AksSysManagerExt.cpp
    AksTopContainer.cpp
    AksTensorBuffer.cpp
)

# Required Packages
find_package(Threads REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(unilog REQUIRED)
find_package(xir REQUIRED)
find_package(vart COMPONENTS runner REQUIRED)

# Add sources to target library
add_library(${PROJECT_NAME} SHARED ${LOG_SRC_LIST})

# Set properties
set_target_properties(${PROJECT_NAME} PROPERTIES
  VERSION "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}"
)

message(STATUS "Python   Includes: ${PYTHON_INCLUDE_DIR}")
message(STATUS "Python3  Includes: ${Python3_INCLUDE_DIRS}")
message(STATUS "Pybind11 Includes: ${pybind11_INCLUDE_DIR}")
message(STATUS "Pybind11 version : ${pybind11_VERSION}")
message(STATUS "Unilog Includes  : ${unilog_INCLUDE_DIRS}")
message(STATUS "VART Includes    : ${vart_INCLUDE_DIRS}")
message(STATUS "XIR Includes     : ${xir_INCLUDE_DIRS}")

# Required Includes
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/>
  $<INSTALL_INTERFACE:aks>
  PRIVATE ${Python3_INCLUDE_DIRS}
  PRIVATE ${pybind11_INCLUDE_DIR}
)

# Required Libs
target_link_libraries(${PROJECT_NAME}
  PRIVATE ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY}
  PRIVATE ${CMAKE_DL_LIBS}
  PRIVATE ${CMAKE_THREAD_LIBS_INIT}
  PRIVATE ${PYTHON_LIBRARIES}
  PRIVATE unilog::unilog
  PRIVATE xir::xir
  PRIVATE vart::runner
)

# Installation properties: Libs
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}-targets
  RUNTIME DESTINATION ${INSTALL_BIN_DIR}
  LIBRARY DESTINATION lib
)
# Installation properties: Includes
install(FILES
  ${CMAKE_SOURCE_DIR}/aks/AksKernelBase.h
  ${CMAKE_SOURCE_DIR}/aks/AksLogger.h
  ${CMAKE_SOURCE_DIR}/aks/AksNodeParams.h
  ${CMAKE_SOURCE_DIR}/aks/AksSysManagerExt.h
  ${CMAKE_SOURCE_DIR}/aks/AksTensorBuffer.h
  ${CMAKE_SOURCE_DIR}/aks/AksBatchTensorBuffer.h
  DESTINATION include/aks
)
# Installation properties: CMake config
install(
  EXPORT ${PROJECT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION share/cmake/${PROJECT_NAME}
)
