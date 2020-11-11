cmake_minimum_required(VERSION 3.12)

project(DPUCADX8GRunner VERSION 1.3.0)

add_library(xfdnn SHARED IMPORTED)

if (NOT "${AKS_INSTALL_PREFIX}" STREQUAL "")
  message(STATUS "AKS Install Prefix: ${AKS_INSTALL_PREFIX}")
  find_package(aks REQUIRED
    PATHS ${AKS_INSTALL_PREFIX}
    NO_DEFAULT_PATH
  )
  set_target_properties(xfdnn PROPERTIES
    IMPORTED_LOCATION "$ENV{VAI_ALVEO_ROOT}/vai/dpuv1/rt/xdnn_cpp/lib/libxfdnn.so"
    INTERFACE_INCLUDE_DIRECTORIES "$ENV{VAI_ALVEO_ROOT}/vai/dpuv1/rt/vitis/include"
  )
else()
  find_package(aks REQUIRED
  )
  set_target_properties(xfdnn PROPERTIES
    IMPORTED_LOCATION "$ENV{CONDA_PREFIX}/lib/libxfdnn.so"
    INTERFACE_INCLUDE_DIRECTORIES "$ENV{CONDA_PREFIX}/lib/python3.6/site-packages/vai/dpuv1/rt/vitis/include"
  )
endif()
message(STATUS "AKS Includes: ${aks_INCLUDE_DIRS}")

add_library (${PROJECT_NAME} SHARED
  AksDPUv1Runner.cpp
)

set_target_properties (${PROJECT_NAME} PROPERTIES
  VERSION   "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}"
)

target_include_directories (${PROJECT_NAME}
  PRIVATE ${aks_INCLUDE_DIRS}
)

target_link_libraries (${PROJECT_NAME}
 PRIVATE xfdnn
)
