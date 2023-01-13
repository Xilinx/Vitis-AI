ARG VAI_BASE=ubuntu:18.04
#nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04
FROM  $VAI_BASE
#From ubuntu:18.04

SHELL ["/bin/bash", "-c"]
ENV TZ=America/Denver
ENV VAI_ROOT=/opt/vitis_ai
ENV VAI_HOME=/vitis_ai_home
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ARG DOCKER_TYPE 
ARG TARGET_FRAMEWORK

WORKDIR /workspace
ADD ./common/ .  
RUN bash ./install_base.sh ${DOCKER_TYPE} ${TARGET_FRAMEWORK}

#install Conda
USER vitis-ai-user
RUN bash ./install_conda.sh 
