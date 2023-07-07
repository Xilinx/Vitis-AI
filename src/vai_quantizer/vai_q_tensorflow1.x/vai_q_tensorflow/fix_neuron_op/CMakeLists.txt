cmake_minimum_required(VERSION 3.0)

project(fix_neuron_op)

# get tensorflow compile info, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()), end='')" OUTPUT_VARIABLE Tensorflow_LINK_FLAGS)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()).replace('ABI=0', 'ABI=0'), end='')" OUTPUT_VARIABLE Tensorflow_COMPILE_FLAGS)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_lib())" OUTPUT_VARIABLE TF_LIBRARY_PATH)


## STRIP space
string(STRIP "${Tensorflow_INCLUDE_DIRS}" Tensorflow_INCLUDE_DIRS)
string(STRIP "${TF_LIBRARY_PATH}" TF_LIBRARY_PATH)

message("tensorflow library path dir is  ${TF_LIBRARY_PATH}")
message("tensorflow include dir is  ${Tensorflow_INCLUDE_DIRS}")
message("tensorflow link flag is  ${Tensorflow_LINK_FLAGS}")
message("tensorflow compile flag is  ${Tensorflow_COMPILE_FLAGS}")


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed -Wl,-rpath='$ORIGIN/':'$ORIGIN/../../tensorflow_core/' ${Tensorflow_LINK_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")

option(GPU "option for GPU" ON)
if (GPU)
  add_definitions(-DBUILD_GPU)
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-10.0/")
  # set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.2/")
  find_package(CUDA)
  MESSAGE(STATUS "cuda found ${CUDA_FOUND}")
  option(WITH_GPU "compile vai_q_tensorflo with cuda:" ${CUDA_FOUND})

  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 ${Tensorflow_COMPILE_FLAGS} -D GOOGLE_CUDA=1 -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr" )
  
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
      #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")
  ENDIF()
endif()

include_directories("./kernels/")
include_directories(${Tensorflow_INCLUDE_DIRS})
include_directories(${Tensorflow_INCLUDE_DIRS}/third_party/gpus/cuda/include)

# set link dir
link_directories(${TF_LIBRARY_PATH})

message("fix_neuron build")
message("CUDA_NVCC_FLAGS is  ${CUDA_NVCC_FLAGS}")
message("CMAKE_CXX_FLAGS is  ${CMAKE_CXX_FLAGS}")


set(op_name "fix_neuron")

if(WITH_GPU)
  MESSAGE(STATUS "Building shared library with GPU support")

  ## let CUDA_ADD_LIBRARY treat *.cu.cc as *.cu file, then will invoke nvcc to compile
  set_source_files_properties(./kernels/${op_name}_ops_gpu.cu.cc PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
  CUDA_ADD_LIBRARY(fix_neuron_op_cu SHARED ./kernels/${op_name}_ops_gpu.cu.cc) 
  TARGET_LINK_LIBRARIES(fix_neuron_op_cu ${CUDA_curand_LIBRARY})
  set_target_properties(fix_neuron_op_cu PROPERTIES PREFIX "")
  
  add_library(fix_neuron_op SHARED ./kernels/${op_name}_ops.cc ./kernels/${op_name}_kernel.cc ./ops/${op_name}_ops.cc )
  set_target_properties(fix_neuron_op PROPERTIES PREFIX "")
  set_target_properties(fix_neuron_op PROPERTIES COMPILE_FLAGS "-DGOOGLE_CUDA")
  
  # target_link_libraries(fix_neuron_op LINK_PUBLIC fix_neuron_op_cu ${Tensorflow_LINK_FLAGS})
  target_link_libraries(fix_neuron_op PUBLIC fix_neuron_op_cu ${Tensorflow_LINK_FLAGS})
ELSE()
  MESSAGE(STATUS "Building shared library without GPU support")
  add_library(fix_neuron_op SHARED ./kernels/${op_name}_ops.cc ./kernels/${op_name}_kernel.cc ./ops/${op_name}_ops.cc )
  set_target_properties(fix_neuron_op PROPERTIES PREFIX "")
  target_link_libraries(fix_neuron_op PUBLIC ${Tensorflow_LINK_FLAGS})
endif()
