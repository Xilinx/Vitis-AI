# Copyright 2019 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# base image
FROM ubuntu:18.04
# author
MAINTAINER samshin

# install packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    checkinstall \
    autoconf \
    automake \
    libtool \
    curl \
    make \
    g++ \
    gcc \
    ca-certificates \
    git \
    wget \
    vim \
    libssl-dev \
    libboost-all-dev \
    libgoogle-glog-dev \
    libreadline-gplv2-dev \
    libncursesw5-dev \
    libssl-dev \
    libsqlite3-dev \
    libjson-c-dev \
    tk-dev \
    libgdbm-dev \
    libc6-dev \
    libbz2-dev \
    zlib1g-dev \
    libffi-dev \
    graphviz \
    python3-pip \
    python3-setuptools\
    && rm -rf /var/lib/apt/lists/*

# create workspace dir
RUN mkdir /root/workspace

# install cmake 3.16
RUN mkdir /tmp/cmake
RUN cd /tmp/cmake
RUN wget -c https://github.com/Kitware/CMake/archive/v3.16.5.tar.gz -O cmake.3.16.5.tar.gz
RUN tar -zxvf cmake.3.16.5.tar.gz
WORKDIR "/tmp/cmake/CMake.3.16.5"
CMD ./bootstrap
CMD make
CMD make install
RUN cat "export PATH=/usr/local/bin:$PATH" >> /root/.bashrc
WORKDIR "/root/workspace/"
RUN rm -r /tmp/cmake/

# install protobuf
WORKDIR "/root/workspace/"
RUN git clone https://github.com/google/protobuf.git
WORKDIR "/root/workspace/protobuf"
RUN git submodule update --init --recursive
RUN ./autogen.sh
RUN ./configure
RUN make
RUN make check
RUN make install
RUN ldconfig
WORKDIR "/root/workspace/"
RUN rm -rf /root/workspace/protobuf

# Prepare to build
RUN mkdir /tmp/Python37
WORKDIR /tmp/Python37

# Pull down Python 3.7, build, and install
RUN wget https://www.python.org/ftp/python/3.7.0/Python-3.7.0.tar.xz --no-check-certificate
RUN tar xvf Python-3.7.0.tar.xz
WORKDIR /tmp/Python37/Python-3.7.0
RUN ./configure
RUN make altinstall
RUN cd /usr/local/bin \
    && ln -s python3.7 python \
    && pip3.7 install --upgrade pip
WORKDIR /root/
RUN rm -r /tmp/Python37/

# install setuptools
CMD pip install -U setuptools black pylint wheel jupyter rope

# install python packages required by xnnc4xir
CMD pip install -U marshmallow tqdm numpy graphviz protobuf h5py==2.10.0

# install python packages required by pyxir
CMD pip install -U pybind11
