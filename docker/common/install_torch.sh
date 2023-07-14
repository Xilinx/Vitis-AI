#!/bin/bash                                                                                                                                                                     
  
set -ex


if [[ ${VAI_CONDA_CHANNEL} =~ .*"tar.gz" ]]; then 
       cd /scratch/;

       wget -O conda-channel.tar.gz --progress=dot:mega ${VAI_CONDA_CHANNEL}; 
       tar -xzvf conda-channel.tar.gz; 
       export VAI_CONDA_CHANNEL=file:///scratch/conda-channel; 
fi
sudo chmod -R 777 /scratch/
sudo ln -s /opt/conda $VAI_ROOT/conda;

. $VAI_ROOT/conda/etc/profile.d/conda.sh \
    && sudo mkdir -p $VAI_ROOT/conda/pkgs && sudo chmod 777 $VAI_ROOT/conda/pkgs \
    && python3 -m pip install --upgrade pip wheel setuptools \
    && sudo  conda config --env --remove-key channels || true  \
    && sudo conda config --env --append channels ${VAI_CONDA_CHANNEL} 
#&& mamba update --force-reinstall --no-deps -n base pytorch_nndct_rocm -c conda-forge \
torchvision_cmd=" pip install torchvision==0.14.1 "
if  [[ ${DOCKER_TYPE} == 'gpu' ]]; then
    torchvision_cmd=" pip install --force-reinstall  torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 "
elif [[ ${DOCKER_TYPE} == 'rocm' ]]; then
   torchvision_cmd=" pip install torchvision==0.14.1+rocm5.2 --extra-index-url https://download.pytorch.org/whl/rocm5.2 "
else
    torchvision_cmd=" pip install torchvision==0.14.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu "

fi

if [[ ${DOCKER_TYPE} == 'rocm' ]]; then
   sudo  ln -s /opt/conda $VAI_ROOT/conda|| true
    
    sudo conda update conda  -y --force-reinstall -c conda-forge  \
    && sudo conda  remove -y -n base --force numpy numpy-base \
    && sudo conda env update -v -f /scratch/${DOCKER_TYPE}_conda/vitis-ai-pytorch.yml \
    && sudo conda install -y -n base pytorch_nndct_rocm -c ${VAI_CONDA_CHANNEL} -c conda-forge  \
    && sudo conda clean -y --force-pkgs-dirs \
    && sudo mkdir -p $VAI_ROOT/compiler \
    && sudo conda config --env --remove-key channels  \
    && sudo cp -r /opt/conda/lib/python3.8/site-packages/vaic/arch $VAI_ROOT/compiler/arch \
    && sudo rm -fr ~/.cache  \
    && sudo rm -fr /scratch/*
else
    . $VAI_ROOT/conda/etc/profile.d/conda.sh \
    && mkdir -p $VAI_ROOT/conda/pkgs \
    && python3 -m pip install --upgrade pip wheel setuptools \
    && conda config --env --remove-key channels || true  \
    && conda config --env --append channels ${VAI_CONDA_CHANNEL} 

    mamba env create -v -f /scratch/${DOCKER_TYPE}_conda/vitis-ai-pytorch.yml \
        && conda activate vitis-ai-pytorch \
        && conda config --show channels \
        && pip install --force-reinstall scipy numpy==1.22 protobuf==3.20.3 tensorboard graphviz==0.19.1 imageio scikit-image  natsort nibabel easydict yacs fire numba loguru \
        &&  mkdir -p $VAI_ROOT/compiler \
        && conda activate vitis-ai-pytorch \
        && $torchvision_cmd \
        && cp -r $CONDA_PREFIX/lib/python3.8/site-packages/vaic/arch $VAI_ROOT/compiler/arch \
        && conda clean -y --force-pkgs-dirs \
        && conda config --env --remove-key channels  \
        && rm -fr ~/.cache 
fi
