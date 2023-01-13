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
VAI_ROOT=${VAI_ROOT:-/opt/vitis_ai} 
mkdir -p ${VAI_ROOT}/

if [[ ${DOCKER_TYPE} == 'rocm' ]]; then
    if [[ -z $VAI_ROOT/conda  ]]; then
        sudo ln -s /opt/conda $VAI_ROOT/conda;
    fi
    
    command_opt=" sudo mamba env update -v -f /scratch/${DOCKER_TYPE}_conda/vitis-ai-optimizer_pytorch.yml";
    . $VAI_ROOT/conda/etc/profile.d/conda.sh \
    && sudo  mkdir -p $VAI_ROOT/conda/pkgs \
    && sudo conda update conda -y --force-reinstall -c conda-forge -c anaconda \
    && sudo conda install -c conda-forge mamba conda-build \
    && sudo python3 -m pip install --upgrade pip wheel setuptools requests \
    && sudo conda config --env --remove-key channels || true \
    && sudo conda config --env --append channels ${VAI_CONDA_CHANNEL} \
    && sudo conda config --remove channels defaults || true \
    && ${command_opt} \
    && sudo conda clean -y --force-pkgs-dirs \
    && sudo rm -fr ~/.cache \
    && sudo rm -fr /scratch/* \
    && sudo  conda config --env --remove-key channels

else
    command_opt=" mamba env create -v -f /scratch/${DOCKER_TYPE}_conda/vitis-ai-optimizer_pytorch.yml ";
    . $VAI_ROOT/conda/etc/profile.d/conda.sh \
    && mkdir -p $VAI_ROOT/conda/pkgs \
    && conda install -c conda-forge mamba conda-build \
    && python3 -m pip install --upgrade pip wheel setuptools requests \
    && conda config --env --remove-key channels || true \
    && conda config --env --append channels ${VAI_CONDA_CHANNEL} \
    && conda config --remove channels defaults || true \
    && ${command_opt} \
    && conda clean -y --force-pkgs-dirs \
    && rm -fr ~/.cache \
    && sudo rm -fr /scratch/* \
    && conda config --env --remove-key channels

fi

