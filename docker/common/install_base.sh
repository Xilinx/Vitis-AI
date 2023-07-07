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
if [[  "$os_version" == "18.04" ]];then

add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get install -y \
        gcc-8 \
        g++-8 \
        gcc-9 \
        g++-9 
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
    && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ bionic main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null \
    && apt-get update -y \
    && apt-get install -y \
        cmake=3.16.0-0kitware1 \
        cmake-data=3.16.0-0kitware1 \
        kitware-archive-keyring \
    && rm -fr /etc/apt/sources.list.d/kitware.list

apt-get install -y ffmpeg \
    && cd /usr/src/gtest \
    && mkdir -p build \
    && cd build \
    && cmake .. \
    && make \
    && make install
  
cd /tmp \
    && wget --progress=dot:mega -O glog.0.4.0.tar.gz https://codeload.github.com/google/glog/tar.gz/v0.4.0 \
    && tar -xvf glog.0.4.0.tar.gz \
    && cd glog-0.4.0 \
    && ./autogen.sh \
    && mkdir build \
    && cd build \
    && cmake -DBUILD_SHARED_LIBS=ON .. \
    && make -j \
    && make install 

cd /tmp; wget --progress=dot:mega https://codeload.github.com/google/protobuf/zip/v3.4.0 \
    && unzip v3.4.0 \
    && cd protobuf-3.4.0 \
    && ./autogen.sh \
    && ./configure \
    && make -j \
    && make install \
    && ldconfig \
    && cd /tmp; wget --progress=dot:mega https://github.com/gflags/gflags/archive/v2.2.2.tar.gz \
    && tar xvf v2.2.2.tar.gz \
    && cd gflags-2.2.2 \
    && mkdir build \
    && cd build \
    && cmake -DBUILD_SHARED_LIBS=ON .. \
    && make -j \
    && make install 

add-apt-repository ppa:timsc/opencv-3.4 \
    && apt-get update \
    && apt install -y\
        libcv-dev=3.4.1-bionic1.1 \
        libcv3.4=3.4.1-bionic1.1 \
        libopencv-calib3d-dev=3.4.1-bionic1.1  \
        libopencv-calib3d3.4-dbg=3.4.1-bionic1.1 \
        libopencv-calib3d3.4=3.4.1-bionic1.1 \
        libopencv-core-dev=3.4.1-bionic1.1 \
        libopencv-core3.4=3.4.1-bionic1.1 \
        libopencv-dev=3.4.1-bionic1.1 \
        libopencv-dnn-dev=3.4.1-bionic1.1 \
        libopencv-dnn3.4-dbg=3.4.1-bionic1.1 \
        libopencv-dnn3.4=3.4.1-bionic1.1 \
        libopencv-features2d-dev=3.4.1-bionic1.1 \
        libopencv-features2d3.4-dbg=3.4.1-bionic1.1 \
        libopencv-features2d3.4=3.4.1-bionic1.1 \
        libopencv-flann-dev=3.4.1-bionic1.1 \
        libopencv-flann3.4-dbg=3.4.1-bionic1.1 \
        libopencv-flann3.4=3.4.1-bionic1.1 \
        libopencv-highgui-dev=3.4.1-bionic1.1 \
        libopencv-highgui3.4-dbg=3.4.1-bionic1.1 \
        libopencv-highgui3.4=3.4.1-bionic1.1 \
        libopencv-imgcodecs-dev=3.4.1-bionic1.1 \
        libopencv-imgcodecs3.4-dbg=3.4.1-bionic1.1 \
        libopencv-imgcodecs3.4=3.4.1-bionic1.1 \
        libopencv-imgproc-dev=3.4.1-bionic1.1 \
        libopencv-imgproc3.4-dbg=3.4.1-bionic1.1 \
        libopencv-imgproc3.4=3.4.1-bionic1.1 \
        libopencv-ml-dev=3.4.1-bionic1.1 \
        libopencv-ml3.4-dbg=3.4.1-bionic1.1 \
        libopencv-ml3.4=3.4.1-bionic1.1 \
        libopencv-objdetect-dev=3.4.1-bionic1.1 \
        libopencv-objdetect3.4-dbg=3.4.1-bionic1.1 \
        libopencv-objdetect3.4=3.4.1-bionic1.1 \
        libopencv-photo-dev=3.4.1-bionic1.1 \
        libopencv-photo3.4-dbg=3.4.1-bionic1.1 \
        libopencv-photo3.4=3.4.1-bionic1.1 \
        libopencv-shape-dev=3.4.1-bionic1.1 \
        libopencv-shape3.4-dbg=3.4.1-bionic1.1 \
        libopencv-shape3.4=3.4.1-bionic1.1 \
        libopencv-stitching-dev=3.4.1-bionic1.1 \
        libopencv-stitching3.4-dbg=3.4.1-bionic1.1 \
        libopencv-stitching3.4=3.4.1-bionic1.1 \
        libopencv-superres-dev=3.4.1-bionic1.1 \
        libopencv-superres3.4-dbg=3.4.1-bionic1.1 \
        libopencv-superres3.4=3.4.1-bionic1.1 \
        libopencv-ts-dev=3.4.1-bionic1.1 \
        libopencv-video-dev=3.4.1-bionic1.1 \
        libopencv-video3.4-dbg=3.4.1-bionic1.1 \
        libopencv-video3.4=3.4.1-bionic1.1 \
        libopencv-videoio-dev=3.4.1-bionic1.1 \
        libopencv-videoio3.4-dbg=3.4.1-bionic1.1 \
        libopencv-videoio3.4=3.4.1-bionic1.1 \
        libopencv-videostab-dev=3.4.1-bionic1.1 \
        libopencv-videostab3.4-dbg=3.4.1-bionic1.1 \
        libopencv-videostab3.4=3.4.1-bionic1.1 \
        libopencv-viz-dev=3.4.1-bionic1.1 \
        libopencv-viz3.4-dbg=3.4.1-bionic1.1 \
        libopencv-viz3.4=3.4.1-bionic1.1 \
        libopencv3.4-java=3.4.1-bionic1.1 \
        opencv-data=3.4.1-bionic1.1 \
        opencv-doc=3.4.1-bionic1.1 \
        python-opencv=3.4.1-bionic1.1 


cd /tmp; git clone https://github.com/pybind/pybind11.git \
    && cd pybind11 \
    && git checkout v2.5.0 \
    && mkdir build \
    && cd build \
    && cmake -DPYBIND11_TEST=OFF .. \
    && make \
    && make install \
    && chmod 777 /usr/lib/python3/dist-packages

update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-9 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-9 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 80 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-8 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-8 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-7 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-7

else
  apt-get update -y  && apt-get  install -y pybind11-dev python3-pybind11 libopencv-dev gcc-9 gcc-10  g++-9 g++-10 \
      libprotobuf-c-dev  protobuf-compiler python-protobuf swig

  update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 10 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-9 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-9 \
  && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 90 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-10 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-10
fi

if [[ $? -ne 0 ]];then
  exit 1
fi


### for all ubuntu18/20


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

  if [[ "$VAI_BASE" == *"22.04"* ]]; then
  # Install common dependencies
    apt-get install -y g++-11
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 30
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 30
    update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-11 30

    # https://www.spinics.net/lists/libreoffice/msg07549.html
  fi

  # Cleanup package manager
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

install_centos() {
  # Need EPEL for many packages we depend on.
  # See http://fedoraproject.org/wiki/EPEL
  yum --enablerepo=extras install -y epel-release

  ccache_deps="asciidoc docbook-dtds docbook-style-xsl libxslt"
  numpy_deps="gcc-gfortran"
  # Note: protobuf-c-{compiler,devel} on CentOS are too old to be used
  # for Caffe2. That said, we still install them to make sure the build
  # system opts to build/use protoc and libprotobuf from third-party.
  yum install -y \
    $ccache_deps \
    $numpy_deps \
    autoconf \
    automake \
    bzip2 \
    cmake \
    cmake3 \
    curl \
    gcc \
    gcc-c++ \
    gflags-devel \
    git \
    glibc-devel \
    glibc-headers \
    glog-devel \
    hiredis-devel \
    libstdc++-devel \
    libsndfile-devel \
    make \
    opencv-devel \
    sudo \
    wget \
    vim

  # Cleanup
  yum clean all
  rm -rf /var/cache/yum
  rm -rf /var/lib/yum/yumdb
  rm -rf /var/lib/yum/history
}

# Install base packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
DOCKER_TYPE=$1  
TARGET_FRAMEWORK=$2

case "$ID" in
  ubuntu)
    install_ubuntu
    ;;
  centos)
    install_centos
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac
