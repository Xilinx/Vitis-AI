# Copyright 2021 Xilinx Inc.
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

# docker build . --build-arg BASE_IMAGE=${BASE_IMAGE}  -t ${DOCKER_IMG_NAME}
ARG BASE_IMAGE=xilinx/vitis-ai-cpu:latest
FROM $BASE_IMAGE

ARG DATE_WEGO
ENV DATE_WEGO=$DATE_WEGO
ARG GIT_HASH_WEGO
ENV GIT_HASH_WEGO=$GIT_HASH_WEGO

ARG VAI_CONDA_CHANNEL_WEGO="file:///scratch/conda-channel"
ENV VAI_CONDA_CHANNEL_WEGO=$VAI_CONDA_CHANNEL_WEGO
ARG CONDA_PACKAGE_PREFIX="https://www.xilinx.com/bin/public/openDownload?filename=conda-channel-wego-2.0.tar"

ADD --chown=vitis-ai-user:vitis-ai-group dockerfiles /scratch/
RUN cat /scratch/banner_wego.sh >> /etc/banner.sh

RUN cd /scratch ;wget -O conda-channel.tar  ${CONDA_PACKAGE_PREFIX} \
    && tar -xvf conda-channel.tar \
    && . $VAI_ROOT/conda/etc/profile.d/conda.sh \
    && mamba create -n vitis-ai-wego-tf1  python=3.6   vai_wego_tensorflow   -c ${VAI_CONDA_CHANNEL_WEGO} -c conda-forge -c anaconda  --override-channels  \
    && rm -fr /scratch/* \
    && rm -fr $VAI_ROOT/conda/pkgs

RUN echo $DATE_WEGO > /etc/BUILD_DATE.txt \
    && echo $GIT_HASH_WEGO > /etc/GIT_HASH.txt \
    && sed -i "s/export\ BUILD_DATE.*/export\ BUILD_DATE=\"${DATE_WEGO}\"/g" ~vitis-ai-user/.bashrc \
    && sed -i "s/export\ BUILD_DATE.*/export\ BUILD_DATE=\"${DATE_WEGO}\"/g" /root/.bashrc \
    && sed -i "s/export\ GIT_HASH.*/export\ GIT_HASH=${GIT_HASH_WEGO}/g" ~vitis-ai-user/.bashrc \
    && sed -i "s/export\ GIT_HASH.*/export\ GIT_HASH=${GIT_HASH_WEGO}/g" /root/.bashrc \
    && sed -i "/export\ BUILD_DATE/ a export GIT_HASH=`cat /etc/GIT_HASH.txt`" /etc/bash.bashrc
