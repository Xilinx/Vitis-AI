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
sudo mkdir $VAI_ROOT/compiler
if [[ ${DOCKER_TYPE} == "cpu"  ]]; then

. $VAI_ROOT/conda/etc/profile.d/conda.sh \
    && mkdir -p  $VAI_ROOT/conda/pkgs \
    && python3 -m pip install --upgrade pip wheel setuptools \
    && conda config --env --append channels ${VAI_WEGO_CONDA_CHANNEL}/wegotf1 \
    && conda config --remove channels defaults || true  \
    && cat ~/.condarc \
    && mamba create  -n  vitis-ai-tensorflow  vai_q_tensorflow  tensorflow-onnx -c  ${VAI_CONDA_CHANNEL} -c conda-forge -c anaconda \
    && conda activate vitis-ai-tensorflow \
    && pip install -r /scratch/pip_requirements.txt keras==2.8 \
    && pip install pycocotools scikit-image tqdm easydict \
    &&  pip uninstall -y tensorflow protobuf \
    && pip uninstall -y h5py \
    && pip uninstall -y h5py \
    && mamba install -y --override-channels --force-reinstall protobuf h5py=2.10.0 -c conda-forge -c anaconda \
    && mamba env update -f /scratch/${DOCKER_TYPE}_conda/vitis-ai-tensorflow.yml \
    && conda config --env --remove-key channels \
    && conda clean -y --force-pkgs-dirs \
    && rm -fr ~/.cache  \
    && sudo rm -fr /scratch/* \
    && sudo cp -r $CONDA_PREFIX/lib/python3.6/site-packages/vaic/arch $VAI_ROOT/compiler/arch
else
    . $VAI_ROOT/conda/etc/profile.d/conda.sh \
    && mkdir -p  $VAI_ROOT/conda/pkgs \
    && python3 -m pip install --upgrade pip wheel setuptools \
    && conda config --env --remove-key channels \
    && conda config --env --add channels ${VAI_WEGO_CONDA_CHANNEL}/wegotf1 \
    && conda config --remove channels defaults  || true \
    && cat ~/.condarc \
    && mamba env create -v -f /scratch/${DOCKER_TYPE}_conda/vitis-ai-tensorflow.yml \
    && conda activate vitis-ai-tensorflow \
    && mamba install vai_q_tensorflow_gpu -c ${VAI_WEGO_CONDA_CHANNEL}/wegotf1 -c conda-forge -c anaconda \
    && pip install -r /scratch/pip_requirements.txt  \
    && pip install pycocotools scikit-image tqdm easydict \
    && pip uninstall -y protobuf \
    && pip uninstall -y h5py \
    && pip uninstall -y h5py \
    && mamba install keras pydot pyyaml jupyter ipywidgets dill pytest scikit-learn pandas matplotlib pillow -c conda-forge -c anaconda \
    && mamba install -y --override-channels --force-reinstall h5py=2.10.0 -c conda-forge -c anaconda \
    && conda install --force-reinstall --freeze-installed -y  -n vitis-ai-tensorflow  protobuf \
    && conda clean -y --force-pkgs-dirs \
    && rm -fr ~/.cache \
    && sudo rm -fr /scratch/* \
    &&  conda config --env --remove-key channels \
    && sudo mkdir -p $VAI_ROOT/compiler \
    && conda activate vitis-ai-tensorflow \
    && sudo cp -r $CONDA_PREFIX/lib/python3.6/site-packages/vaic/arch $VAI_ROOT/compiler/arch

fi
