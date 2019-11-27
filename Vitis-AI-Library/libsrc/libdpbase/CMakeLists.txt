cmake_minimum_required(VERSION 3.5)
project(dpbase VERSION 1.0.0 LANGUAGES C CXX)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SYSROOT}/usr/share/cmake/")
include_directories(${CMAKE_SOURCE_DIR}/include)
set(OpenCV_LIBS opencv_core opencv_video opencv_videoio opencv_imgproc opencv_imgcodecs opencv_highgui)
if(CMAKE_SYSTEM_NAME STREQUAL "QNX")
  ## TODO: fix hard coded directory
  link_directories(/home/administrator/workspace/qtbase/lib/)
  list(APPEND OpenCV_LIBS Qt5Widgets Qt5Test Qt5Concurrent Qt5Gui Qt5Core)
else()
  list(APPEND OpenCV_LIBS pthread)
endif()
include(XilinxCommon)
include(XilinxVersion)

include_directories(${CMAKE_BINARY_DIR})
add_library(${PROJECT_NAME}
  include/xilinx/ai/dpu_task.hpp
  include/xilinx/ai/configurable_dpu_task.hpp
  src/dpu_task.cpp
  src/dpu_task_imp.hpp
  src/dpu_task_imp.cpp
  src/banner.hpp
  src/banner.cpp
  src/configurable_dpu_task.cpp
  src/configurable_dpu_task_imp.hpp
  src/configurable_dpu_task_imp.cpp
  )


target_link_libraries(${PROJECT_NAME} dpmath dpcommon dpproto glog protobuf ${OpenCV_LIBS}
  vitis_dpu # defined in vitis/runtime-api
  )
set_target_properties(${PROJECT_NAME} PROPERTIES
  VERSION "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}")

install(FILES
  include/xilinx/ai/configurable_dpu_task.hpp
  include/xilinx/ai/dpu_task.hpp
  DESTINATION include/xilinx/ai)


install(TARGETS ${PROJECT_NAME} DESTINATION lib)


install(FILES
  src/banner.hpp
  DESTINATION include/xilinx/ai)



add_executable(test_it test/test_it.cpp)
target_link_libraries(test_it ${PROJECT_NAME})

add_executable(test_base test/test_base.cpp)
target_link_libraries(test_base ${PROJECT_NAME})

add_executable(test_dpu_task test/test_dpu_task.cpp)
target_link_libraries(test_dpu_task ${PROJECT_NAME})
# TODO ADD TESTS
