cmake_minimum_required(VERSION 3.12)

project(DPUCZDX8GRunner VERSION 1.3.0)

set(CMAKE_CXX_STANDARD 17)

if (NOT "${AKS_INSTALL_PREFIX}" STREQUAL "")
  message(STATUS "AKS Install Prefix: ${AKS_INSTALL_PREFIX}")
  find_package(aks REQUIRED
    PATHS ${AKS_INSTALL_PREFIX}
    NO_DEFAULT_PATH
  )
else()
  find_package(aks REQUIRED
  )
endif()
message(STATUS "AKS Includes: ${aks_INCLUDE_DIRS}")

add_library (${PROJECT_NAME} SHARED
  AksDPUCZDX8GRunner.cpp
  common.cpp
)

set_target_properties (${PROJECT_NAME} PROPERTIES
  VERSION   "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}"
)

target_include_directories (${PROJECT_NAME}
  PRIVATE ${aks_INCLUDE_DIRS}
)

target_link_libraries (${PROJECT_NAME}
  PRIVATE ${aks_LIBRARIES}
  PRIVATE vart-runner
  PRIVATE glog xir unilog
  PRIVATE ${CMAKE_THREAD_LIBS_INIT}
)
