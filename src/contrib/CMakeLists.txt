include_directories(${CMAKE_SOURCE_DIR}/src)

find_package(unilog REQUIRED)
include_directories(${unilog_INCLUDE_DIRS})

aux_source_directory(. CONTRIB_SRCS)
add_library(xir_contrib SHARED ${CONTRIB_SRCS})
target_link_libraries(xir_contrib ${PROJECT_NAME} glog::glog)
