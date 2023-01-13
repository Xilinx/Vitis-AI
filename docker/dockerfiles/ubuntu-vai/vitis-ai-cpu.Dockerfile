#FROM ubuntu:${UBUNTU_VERSION}
ARG VAI_BASE=artifactory.xilinx.com/vitis-ai-docker-master-local/vitis-ai-cpu-conda-base:latest                                                                               
  
From $VAI_BASE
ARG TARGET_FRAMEWORK
ENV TARGET_FRAMEWORK=$TARGET_FRAMEWORK
ARG VAI_CONDA_CHANNEL="file:///scratch/conda-channel"
ENV VAI_CONDA_CHANNEL=$VAI_CONDA_CHANNEL
ARG VERSION
ENV VERSION=$VERSION
ARG DOCKER_TYPE='cpu'
ENV DOCKER_TYPE=$DOCKER_TYPE
ARG GIT_HASH="<blank>"
ENV GIT_HASH=$GIT_HASH
ARG  BUILD_DATE
ENV BUILD_DATE=$BUILD_DATE
ARG XRT_URL=https://www.xilinx.com/bin/public/openDownload?filename=xrt_202120.2.12.427_18.04-amd64-xrt.deb
ENV XRT_URL=$XRT_URL
ARG XRM_URL=https://www.xilinx.com/bin/public/openDownload?filename=xrm_202120.1.3.29_18.04-x86_64.deb
ENV XRM_URL=$XRM_URL
ARG VAI_DEB_CHANNEL=""
ENV VAI_DEB_CHANNEL=$VAI_DEB_CHANNEL
ARG VAI_WEGO_CONDA_CHANNEL="file:///scratch/conda-channel-wego"
ENV VAI_WEGO_CONDA_CHANNEL=$VAI_WEGO_CONDA_CHANNEL


WORKDIR /workspace
ADD ./common/ .  
ADD ./conda /scratch
ADD conda/banner.sh /etc/
ADD conda/${DOCKER_TYPE}_conda/bashrc /etc/bash.bashrc
RUN if [[ -n "${TARGET_FRAMEWORK}" ]]; then  bash ./install_${TARGET_FRAMEWORK}.sh; fi
USER root
RUN mkdir -p ${VAI_ROOT}/conda/pkgs && chmod 777 ${VAI_ROOT}/conda/pkgs && ./install_vairuntime.sh && rm -fr ./*
