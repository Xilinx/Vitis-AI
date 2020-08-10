ARG ARCH=
ARG CUDA=10.0
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG ARCH
ARG CUDA
ARG CUDNN=7.6.2.24-1
ARG CUDNN_MAJOR_VERSION=7
ARG LIB_DIR_PREFIX=x86_64

# Needed for string substitution
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        cuda-cublas-dev-${CUDA/./-} \
        cuda-cudart-dev-${CUDA/./-} \
        cuda-cufft-dev-${CUDA/./-} \
        cuda-curand-dev-${CUDA/./-} \
        cuda-cusolver-dev-${CUDA/./-} \
        cuda-cusparse-dev-${CUDA/./-} \
        libcudnn7=${CUDNN}+cuda${CUDA} \
        libcudnn7-dev=${CUDNN}+cuda${CUDA} \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        wget \
        git \
        && \
    find /usr/local/cuda-${CUDA}/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
    rm /usr/lib/${LIB_DIR_PREFIX}-linux-gnu/libcudnn_static_v7.a

RUN [[ "${ARCH}" = "ppc64le" ]] || { apt-get update && \
        apt-get install -y --no-install-recommends libnvinfer5=5.1.5-1+cuda${CUDA} \
        libnvinfer-dev=5.1.5-1+cuda${CUDA} \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*; }

# Configure the build for our CUDA configuration.
ENV CI_BUILD_PYTHON python
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV TF_NEED_CUDA 1
ENV TF_NEED_TENSORRT 1
ENV TF_CUDA_COMPUTE_CAPABILITIES=3.5,5.2,6.0,6.1,7.0
ENV TF_CUDA_VERSION=${CUDA}
ENV TF_CUDNN_VERSION=${CUDNN_MAJOR_VERSION}
# CACHE_STOP is used to rerun future commands, otherwise cloning tensorflow will be cached and will not pull the most recent version
ARG CACHE_STOP=1
# Check out TensorFlow source code if --build-arg CHECKOUT_TF_SRC=1
ARG CHECKOUT_TF_SRC=0
RUN test "${CHECKOUT_TF_SRC}" -eq 1 && git clone https://github.com/tensorflow/tensorflow.git /tensorflow_src || true

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig
