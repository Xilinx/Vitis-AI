#!/bin/bash                                                                                                                                                                     
  
set -ex
sudo chmod 777 /scratch
# vitis-ai-optimizer_tensorflow
if [[ ${VAI_CONDA_CHANNEL} =~ .*"tar.gz" ]]; then \
       cd /scratch/; \
       wget -O conda-channel.tar.gz --progress=dot:mega ${VAI_CONDA_CHANNEL}; \
       tar -xzvf conda-channel.tar.gz; \
       export VAI_CONDA_CHANNEL=file:///scratch/conda-channel; 
fi
. $VAI_ROOT/conda/etc/profile.d/conda.sh \
    && mkdir  -p $VAI_ROOT/conda/pkgs \
    && sudo chmod -R 777 /scratch \
    && sudo python3 -m pip install --upgrade pip wheel setuptools \
    && conda config --env --remove-key channels \
    && conda config --env --append channels ${VAI_CONDA_CHANNEL} \
    && conda config --remove channels defaults || true \
    && mamba env create -f /scratch/${DOCKER_TYPE}_conda/vitis-ai-optimizer_tensorflow2.yml \
    && conda clean -y --force-pkgs-dirs \
    && sudo rm -fr ~/.cache \
    && sudo rm -fr /scratch/* \
    && conda config --env --remove-key channels 
