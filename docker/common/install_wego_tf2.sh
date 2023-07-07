#!/bin/bash                                                                                                                                                                     
set -ex
if [[ ${VAI_CONDA_CHANNEL} =~ .*"tar.gz" ]]; then \
       cd /scratch/; \
       wget -O conda-channel.tar.gz --progress=dot:mega ${VAI_CONDA_CHANNEL}; \
       tar -xzvf conda-channel.tar.gz; \
       export VAI_CONDA_CHANNEL=file:///scratch/conda-channel; \
       wget -O conda-channel-wego.tar.gz --progress=dot:mega ${VAI_WEGO_CONDA_CHANNEL}; \
       tar -xzvf conda-channel-wego.tar.gz; \
       export VAI_WEGO_CONDA_CHANNEL=file:///scratch/conda-channel-wego; 
fi; 
. $VAI_ROOT/conda/etc/profile.d/conda.sh \
    && mkdir -p $VAI_ROOT/conda/pkgs \
    && mamba install -c conda-forge conda-build \
    && python3 -m pip install --upgrade pip wheel setuptools \
    && conda config --env --append channels ${VAI_WEGO_CONDA_CHANNEL}/wegotf2 \
    && conda config --remove channels defaults || true \
    && cat ~/.condarc \
    && mamba env create -f /scratch/${DOCKER_TYPE}_conda/vitis-ai-wego-tf2.yml \
    && conda config --env --remove-key channels \
    && conda activate vitis-ai-wego-tf2 \
    && mamba install --no-update-deps -y vai_q_tensorflow2  -c  ${VAI_CONDA_CHANNEL} -c conda-forge -c anaconda \
    && conda clean -y --force-pkgs-dirs \
    && rm -fr ~/.cache  \
    && sudo rm -fr /scratch/* \
    && conda config --env --remove-key channels 
