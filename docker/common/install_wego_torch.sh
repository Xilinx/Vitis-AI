#!/bin/bash                                                                                                                                                                     
set -ex

if [[ ${VAI_CONDA_CHANNEL} =~ .*"tar.gz" ]]; then 
       cd /scratch/; 
       wget -O conda-channel.tar.gz --progress=dot:mega ${VAI_CONDA_CHANNEL}; 
       tar -xzvf conda-channel.tar.gz; 
       export VAI_CONDA_CHANNEL=file:///scratch/conda-channel; 
fi
sudo chmod -R 777 /scratch/
. $VAI_ROOT/conda/etc/profile.d/conda.sh \
    && mkdir -p $VAI_ROOT/conda/pkgs \
    && python3 -m pip install --upgrade pip wheel setuptools \
    && conda config --env --append channels ${VAI_CONDA_CHANNEL} \
    && conda config --remove channels defaults  || true \
    && cat ~/.condarc \
    && mamba env create -f /scratch/${DOCKER_TYPE}_conda/vitis-ai-wego-torch.yml \
    && conda clean -y --force-pkgs-dirs \
    && rm -fr ~/.cache  \
    && sudo rm -fr /scratch/* \
    && conda config --env --remove-key channels 
