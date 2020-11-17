cmake_minimum_required(VERSION 3.12)

project(ClassificationPreProcessAccel VERSION 1.3.0)

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

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)

add_library (${PROJECT_NAME} SHARED
  AksPreProcKernel.cpp
  $ENV{VAI_ALVEO_ROOT}/apps/whole_app_acceleration/classification/src/xcl2.cpp
)

set_target_properties (${PROJECT_NAME} PROPERTIES
  VERSION   "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}"
)

target_include_directories (${PROJECT_NAME}
  PRIVATE ${aks_INCLUDE_DIRS}
  PRIVATE ${OpenCV_INCLUDE_DIRS}
  PRIVATE ${OpenCL_INCLUDE_DIRS}
  PRIVATE "$ENV{XILINX_XRT}/include"
  PRIVATE "$ENV{VAI_ALVEO_ROOT}/apps/whole_app_acceleration/classification/src/"
)

target_link_libraries (${PROJECT_NAME}
 PRIVATE ${OpenCV_LIBS}
 PRIVATE ${OpenCL_LIBRARIES}
)
