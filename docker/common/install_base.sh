#!/bin/bash

set -ex

install_ubuntu() {
  # NVIDIA dockers for RC releases use tag names like `11.0-cudnn8-devel-ubuntu18.04-rc`,
  # for this case we will set UBUNTU_VERSION to `18.04-rc` so that the Dockerfile could
  # find the correct image. As a result, here we have to check for
  #   "$UBUNTU_VERSION" == "18.04"*
  # instead of
  #   "$UBUNTU_VERSION" == "18.04"
  # TODO: Remove this once nvidia package repos are back online
  # Comment out nvidia repositories to prevent them from getting apt-get updated, see https://github.com/pytorch/pytorch/issues/74968
  # shellcheck disable=SC2046
  sed -i 's/.*nvidia.*/# &/' $(find /etc/apt/ -type f -name "*.list")
if [[ ${DOCKER_TYPE} =~ .*"rocm"*  && ${TARGET_FRAMEWORK} =~ .*"pytorch"   ]]; then
     echo "using rocm pytorch  imge"
     apt-get update -y \
     && apt-get install -y --no-install-recommends locales

elif [[  ${DOCKER_TYPE} =~ .*"rocm"*  ]];then
    apt-get update  -y \
    && apt-get install -y  wget rccl lsb-release


else

chmod 1777 /tmp \
    && mkdir /scratch \
    && chmod 1777 /scratch \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        apt-transport-https \
        autoconf \
        automake \
        bc \
        build-essential \
        bzip2 \
        ca-certificates \
        curl \
        g++ \
        gdb \
        git \
        gnupg \
        locales \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libjsoncpp-dev \
        libssl-dev \
        libtool \
        libunwind-dev \
        make \
        cmake \
        openssh-client \
        openssl \
        python3 \
        python3-dev \
        python3-minimal \
        python3-numpy \
        python3-opencv \
        python3-pip \
        python3-setuptools \
        python3-venv \
        software-properties-common \
        sudo \
        tree \
        tzdata \
        unzip \
        vim \
        wget \
        yasm \
        zstd \
        libavcodec-dev \
        libavformat-dev \
        libeigen3-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer1.0-dev \
        libgtest-dev \
        libgtk-3-dev \
        libgtk2.0-dev \
        libhdf5-dev \
        libjpeg-dev \
        libopenexr-dev \
        libpng-dev \
        libswscale-dev \
        libtiff-dev \
        libwebp-dev \
        opencl-clhpp-headers \
        opencl-headers \
        pocl-opencl-icd \
        python3-opencv \
        python3-pip \
        python3-setuptools \
        python3-venv \
        software-properties-common \
        sudo \
        tree \
        unzip \
        vim \
        wget \
        yasm \
        zstd \
        libavcodec-dev \
        libavformat-dev \
        libeigen3-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer1.0-dev \
        libgtest-dev \
        libgtk-3-dev \
        libgtk2.0-dev \
        libhdf5-dev \
        libjpeg-dev \
        libopenexr-dev \
        libpng-dev \
        libswscale-dev \
        libtiff-dev \
        libwebp-dev \
        opencl-clhpp-headers \
        opencl-headers \
        ffmpeg \
        pocl-opencl-icd 

fi

os_version=`lsb_release -r -s`
echo "base OS version:${os_version}"
apt-get update -y  && apt-get  install -y pybind11-dev python3-pybind11 libopencv-dev gcc-9 gcc-10  g++-9 g++-10 \
      libprotobuf-c-dev  protobuf-compiler python-protobuf swig

  update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 10 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-9 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-9 \
  && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 90 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-10 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-10

if [[ $? -ne 0 ]];then
  exit 1
fi



pip3 install \
        Flask \
        setuptools \
        wheel

sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && echo "LC_ALL=en_US.UTF-8" >> /etc/environment \
    && echo "LANG=en_US.UTF-8" > /etc/locale.conf \
    && locale-gen en_US.UTF-8 \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8 \
    && dpkg-reconfigure locales \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata



cd /tmp && wget --progress=dot:mega https://github.com/json-c/json-c/archive/json-c-0.15-20200726.tar.gz \
    && tar xvf json-c-0.15-20200726.tar.gz \
    && cd json-c-json-c-0.15-20200726 \
    && mkdir build \
    && cd build \
    && cmake -DBUILD_SHARED_LIBS=ON .. \
    && make -j \
    && make install 





export GOSU_VERSION="1.12"

#COPY dockerfiles/bashrc /etc/bash.bashrc
#RUN chmod a+rwx /etc/bash.bashrc
groupadd vitis-ai-group \
    && useradd --shell /bin/bash -c '' -m -g vitis-ai-group vitis-ai-user \
    && passwd -d vitis-ai-user \
    && usermod -aG sudo vitis-ai-user \
    && echo 'ALL ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && echo 'Defaults        secure_path="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/vitis_ai/conda/bin"' >> /etc/sudoers \
    && curl -sSkLo /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$(dpkg --print-architecture)" \
    && chmod +x /usr/local/bin/gosu
  

mkdir -p ${VAI_ROOT} \
        && chown -R vitis-ai-user:vitis-ai-group ${VAI_ROOT} 

  # Cleanup package manager
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

# Install base packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
DOCKER_TYPE=$1  
TARGET_FRAMEWORK=$2

case "$ID" in
  ubuntu)
    install_ubuntu
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac
