IF (APPLE)
    cmake_minimum_required(VERSION 3.4)
ELSE()
    cmake_minimum_required(VERSION 2.8)
ENDIF()

project(nndct_release)

IF (NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O2")
ENDIF()

IF (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O2")
    add_definitions(-DAPPLE)
ENDIF()

FIND_PACKAGE(CUDA 6.5)
MESSAGE(STATUS "cuda found ${CUDA_FOUND}")

option(WITH_GPU "compile nndct with cuda." ${CUDA_FOUND})
option(WITH_OMP "compile nndct with openmp." ON)

if(WITH_GPU)
	add_definitions(-DNNDCT_ENABLE_GPU)
endif()

if(NOT WITH_OMP)
	add_definitions(-DNNDCT_DISABLE_OMP)
endif()

# need to be at least 30 or __shfl_down in reduce wont compile
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")

IF (CUDA_VERSION GREATER 7.6)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_62,code=sm_62")
ENDIF()

IF (CUDA_VERSION GREATER 8.9)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
ENDIF()

IF (CUDA_VERSION GREATER 9.9)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
ENDIF()

IF (CUDA_VERSION GREATER 10.9)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")
ENDIF()

if (NOT APPLE)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O2 --std=c++14")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fopenmp")
ENDIF()

IF (APPLE)
    EXEC_PROGRAM(uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
    STRING(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
    MESSAGE(STATUS "DARWIN_VERSION=${DARWIN_VERSION}")

    #for el capitain have to use rpath

    IF (DARWIN_VERSION LESS 15)
        set(CMAKE_SKIP_RPATH TRUE)
    ENDIF ()

ELSE()
    #always skip for linux
    set(CMAKE_SKIP_RPATH TRUE)
ENDIF()


IF (WITH_GPU)

    MESSAGE(STATUS "Building shared library with GPU support")
    
    FILE(GLOB_RECURSE CC_LIST ./csrc/*.cc)
    FILE(GLOB_RECURSE CU_LIST ./csrc/*.cu)
    CUDA_ADD_LIBRARY(nndct SHARED ${CC_LIST} ${CU_LIST})

    TARGET_LINK_LIBRARIES(nndct ${CUDA_curand_LIBRARY})

    INSTALL(TARGETS nndct
            RUNTIME DESTINATION "bin"
            LIBRARY DESTINATION "lib"
            ARCHIVE DESTINATION "lib")

ELSE()
    MESSAGE(STATUS "Building shared library with no GPU support")

ENDIF()


