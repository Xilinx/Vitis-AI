cmake_minimum_required(VERSION 3.12)

project(YoloPostProcess VERSION 1.3.0)

add_library(dpuv1utils SHARED IMPORTED)

if (NOT "${AKS_INSTALL_PREFIX}" STREQUAL "")
  message(STATUS "AKS Install Prefix: ${AKS_INSTALL_PREFIX}")
  find_package(aks REQUIRED
    PATHS ${AKS_INSTALL_PREFIX}
    NO_DEFAULT_PATH
  )
  set_target_properties(dpuv1utils PROPERTIES
    IMPORTED_LOCATION "$ENV{VAI_ALVEO_ROOT}/vai/dpuv1/utils/libdpuv1utils.so"
    INTERFACE_INCLUDE_DIRECTORIES "$ENV{VAI_ALVEO_ROOT}/vai/dpuv1/utils/postproc"
  )
else()
  find_package(aks REQUIRED
  )
  set_target_properties(dpuv1utils PROPERTIES
    IMPORTED_LOCATION "$ENV{CONDA_PREFIX}/lib/python3.6/site-packages/vai/dpuv1/utils/libdpuv1utils.so"
    INTERFACE_INCLUDE_DIRECTORIES "$ENV{CONDA_PREFIX}/lib/python3.6/site-packages/vai/dpuv1/utils/postproc/"
  )
endif()
message(STATUS "AKS Includes: ${aks_INCLUDE_DIRS}")

find_package(OpenCV REQUIRED)


add_library (${PROJECT_NAME} SHARED
  AksYoloPostProc.cpp
)

set_target_properties (${PROJECT_NAME} PROPERTIES
  VERSION   "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}"
)

target_include_directories (${PROJECT_NAME}
  PRIVATE ${aks_INCLUDE_DIRS}
  PRIVATE ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries (${PROJECT_NAME}
 PRIVATE dpuv1utils
 PRIVATE ${OpenCV_LIBS}
)
