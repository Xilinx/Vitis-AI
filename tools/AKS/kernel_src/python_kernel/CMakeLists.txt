cmake_minimum_required(VERSION 3.12)

project(PythonKernel VERSION 1.3.0)

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

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

message(STATUS "Python3  Includes: ${Python3_INCLUDE_DIRS}")
message(STATUS "Pybind11 Includes: ${pybind11_INCLUDE_DIR}")

add_library (${PROJECT_NAME} SHARED
  AksPythonKernel.cpp
)

set_target_properties (${PROJECT_NAME} PROPERTIES
  VERSION   "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}"
)

target_include_directories (${PROJECT_NAME}
  PRIVATE ${CMAKE_SOURCE_DIR}
  PRIVATE ${aks_INCLUDE_DIRS}
  PRIVATE ${Python3_INCLUDE_DIRS}
  PRIVATE ${pybind11_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE ${PYTHON_LIBRARIES}
)
