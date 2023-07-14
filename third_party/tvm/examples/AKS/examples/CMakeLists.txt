cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)

project(examples VERSION 1.3.0)

 # Set source files of examples (non-video)
  set (EXAMPLES_IMAGE
    resnet50_dpucadf8h_tvm.cpp
    yolov3_dpucadf8h_tvm.cpp
    resnet50_dpuczdx8g_zcu014_tvm.cpp
  )
  # Set source files of examples (video)
  set (EXAMPLES_VIDEO
    
  )

find_package(xir REQUIRED)
find_package(unilog REQUIRED)
find_package(vart REQUIRED)

# Find Packages
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

execute_process(COMMAND uname -m OUTPUT_VARIABLE arch)
find_package(Threads REQUIRED)
if(${arch} MATCHES ".*x86.*")
  find_package(Boost 1.65.1 EXACT REQUIRED COMPONENTS system filesystem)
else()
  find_package(Boost 1.65.1 REQUIRED COMPONENTS system filesystem)
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgproc video videoio)

# Set output directories for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Compile Sources
foreach( sourcefile ${EXAMPLES_IMAGE} )
  # Generate executable name
  string( REPLACE ".cpp" ".exe" exename ${sourcefile} )

  # Set target
  add_executable( ${exename} ${sourcefile} )

  # Set include dirs
  target_include_directories (${exename}
    PRIVATE ${aks_INCLUDE_DIRS}
    PRIVATE ${Boost_INCLUDE_DIRS}
  )

  # Set libraries to be linked
  target_link_libraries (${exename}
    PRIVATE ${aks_LIBRARIES}
    PRIVATE ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY}
    PRIVATE ${CMAKE_DL_LIBS}
    PRIVATE ${CMAKE_THREAD_LIBS_INIT}
    PRIVATE vart-runner
    PRIVATE vart-util
    PRIVATE xir unilog
  )
  # set (CMAKE_DEBUG_TARGET_PROPERTIES INCLUDE_DIRECTORIES)
endforeach()
