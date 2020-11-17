cmake_minimum_required(VERSION 3.12)

project(examples VERSION 1.3.0)

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

find_package(Threads REQUIRED)
find_package(Boost 1.65.1 EXACT REQUIRED COMPONENTS system filesystem)
find_package(OpenCV REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set (EXAMPLES
  facedetect.cpp
  googlenet.cpp
  googlenet_pp_accel.cpp
  googlenet_resnet50.cpp
  googlenet_tinyyolov3.cpp
  inception_v1_tf.cpp
  resnet50.cpp
  stdyolov2.cpp
  tinyyolov3.cpp
  #tinyyolov3_video.cpp
)

foreach( sourcefile ${EXAMPLES} )
  string( REPLACE ".cpp" ".exe" exename ${sourcefile} )
  add_executable( ${exename} ${sourcefile} )
  target_include_directories (${exename}
    PRIVATE ${aks_INCLUDE_DIRS}
    #PRIVATE ${OpenCV_INCLUDES}
  )
  target_link_libraries (${exename}
    PRIVATE ${aks_LIBRARIES}
    PRIVATE ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY}
    PRIVATE ${CMAKE_DL_LIBS}
    PRIVATE ${CMAKE_THREAD_LIBS_INIT}
    #PRIVATE ${OpenCV_LIBS}
  )
endforeach()
