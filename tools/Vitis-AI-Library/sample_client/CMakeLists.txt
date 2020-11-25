cmake_minimum_required(VERSION 3.12)
project(hello_vitis_ai_lib VERSION 1.0.0 LANGUAGES C CXX)
find_package(vitis_ai_library REQUIRED)
# find_package(OpenCV REQUIRED core videoio) already included by vitis-ai-library
message("OpenCV_LIBS is ${OpenCV_LIBS}")
add_executable(showargs showargs.cpp)
install(TARGETS showargs DESTINATION bin)
