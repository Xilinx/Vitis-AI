cmake_minimum_required(VERSION 3.12)

project(FallDetectionEvaluation VERSION 1.3.0)

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
find_package (Boost 1.65 REQUIRED)

add_library (${PROJECT_NAME} SHARED
  AksFallDetectionEvaluation.cpp
)

set_target_properties (${PROJECT_NAME} PROPERTIES
  VERSION   "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}"
)

target_include_directories (${PROJECT_NAME}
  PRIVATE ${aks_INCLUDE_DIRS}
  PRIVATE ${OpenCV_INCLUDES}
)

target_link_libraries (${PROJECT_NAME}
 PRIVATE ${OpenCV_LIBS}
 PRIVATE ${Boost_LIBRARY}
)
