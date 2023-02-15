#!/bin/bash                                                                                                                                                                     
  
set -ex
sudo chmod 777 /scratch
if [[ ${VAI_CONDA_CHANNEL} =~ .*"tar.gz" ]]; then \
       cd /scratch/; \
       wget -O conda-channel.tar.gz --progress=dot:mega ${VAI_CONDA_CHANNEL}; \
       tar -xzvf conda-channel.tar.gz; \
       export VAI_CONDA_CHANNEL=file:///scratch/conda-channel; \
       wget -O conda-channel-wego.tar.gz --progress=dot:mega ${VAI_WEGO_CONDA_CHANNEL}; \
       tar -xzvf conda-channel-wego.tar.gz; \
       export VAI_WEGO_CONDA_CHANNEL=file:///scratch/conda-channel-wego;
fi;
sudo mkdir -p $VAI_ROOT/compiler 

if [[ ${DOCKER_TYPE} != 'cpu' ]]; then \
    arch_type="_${DOCKER_TYPE}";
else
    arch_type="";

fi

conda_channel="${VAI_WEGO_CONDA_CHANNEL}/wegotf2"

if [[ ${DOCKER_TYPE} == 'rocm' ]]; then \
    tensorflow_ver="tensorflow-${DOCKER_TYPE}==2.10.1.540  keras==2.10";
#    conda_channel="${VAI_CONDA_CHANNEL}"
else
    tensorflow_ver="tensorflow==2.10 keras==2.10";
     conda_channel="${VAI_WEGO_CONDA_CHANNEL}/wegotf2"

fi
if [[ ${DOCKER_TYPE} == 'cpu' ]]; then
   . $VAI_ROOT/conda/etc/profile.d/conda.sh \
    && mkdir -p $VAI_ROOT/conda/pkgs \
    && python3 -m pip install --upgrade pip wheel setuptools \
    && conda config --env --append channels ${VAI_WEGO_CONDA_CHANNEL}/wegotf2 \
    && conda config --remove channels defaults || true \
    && cat ~/.condarc \
    && mamba env create -f /scratch/${DOCKER_TYPE}_conda/vitis-ai-tensorflow2.yml \
    && conda activate vitis-ai-tensorflow2 \
    && mamba install --no-update-deps  vai_q_tensorflow2 pydot pyyaml jupyter ipywidgets \
            dill progressbar2 pytest scikit-learn pandas matplotlib \
            pillow -c ${VAI_WEGO_CONDA_CHANNEL}/wegotf2 -c conda-forge -c defaults \
        && pip install -r /scratch/pip_requirements.txt \
        && pip install pycocotools scikit-image tqdm easydict \
        && pip uninstall -y h5py \
        && pip uninstall -y h5py \
        && mamba install -y --override-channels --force-reinstall h5py=2.10.0 -c conda-forge -c anaconda \
    && pip install --force --no-binary protobuf protobuf==3.19.4 \
    && pip uninstall -y protobuf \
    && pip install protobuf==3.9.2 \
    && pip install --force --no-binary protobuf protobuf==3.19.4 \
    && conda config --env --remove-key channels \
    && conda clean -y --force-pkgs-dirs \
    && sudo cp -r $CONDA_PREFIX/lib/python3.7/site-packages/vaic/arch $VAI_ROOT/compiler/arch \
    && rm -fr ~/.cache  \
    && sudo rm -fr /scratch/* 
elif [[ ${DOCKER_TYPE} == 'rocm' ]]; then
  . $VAI_ROOT/conda/etc/profile.d/conda.sh \
    && mkdir -p $VAI_ROOT/conda/pkgs \
    && sudo python3 -m pip install --upgrade pip wheel setuptools \
    && conda config --env --remove-key channels \
    && conda config --env --append channels ${conda_channel}  \
    && conda config --remove channels defaults || true \
    && mamba env create -f /scratch/${DOCKER_TYPE}_conda/vitis-ai-tensorflow2.yml \
    && conda activate vitis-ai-tensorflow2 \
    && mamba install --no-update-deps -y vai_q_tensorflow2${arch_type} pydot pyyaml jupyter ipywidgets \
            dill progressbar2 pytest scikit-learn pandas matplotlib \
            pillow -c ${conda_channel} -c conda-forge -c defaults \
        && pip install -r /scratch/pip_requirements.txt \
        && pip install pycocotools scikit-image tqdm easydict \
        && pip install --ignore-installed ${tensorflow_ver} \
        && pip install --force --no-binary protobuf protobuf==3.19.4 \
        && pip uninstall -y h5py \
        && pip uninstall -y h5py  \
        && mamba install -y --override-channels --force-reinstall h5py=2.10.0 -c conda-forge -c anaconda \
    && conda clean -y --force-pkgs-dirs \
    && sudo rm -fr ~/.cache \
    && sudo rm -fr /scratch/* \
    && conda config --env --remove-key channels \
    && conda activate vitis-ai-tensorflow2 \
    && sudo mkdir -p $VAI_ROOT/compiler \
    && sudo cp -r $CONDA_PREFIX/lib/python3.7/site-packages/vaic/arch $VAI_ROOT/compiler/arch
else
. $VAI_ROOT/conda/etc/profile.d/conda.sh \
    && mkdir -p $VAI_ROOT/conda/pkgs \
    && sudo python3 -m pip install --upgrade pip wheel setuptools \
    && conda config --env --remove-key channels \
    && conda config --env --append channels ${conda_channel}  \
    && conda config --remove channels defaults || true \
    && mamba env create -f /scratch/${DOCKER_TYPE}_conda/vitis-ai-tensorflow2.yml \
    && conda activate vitis-ai-tensorflow2 \
    && mamba install --no-update-deps -y vai_q_tensorflow2${arch_type} pydot pyyaml jupyter ipywidgets \
            dill progressbar2 pytest scikit-learn pandas matplotlib \
            pillow -c ${conda_channel} -c conda-forge -c defaults \
        && pip install -r /scratch/pip_requirements.txt \
        && pip install pycocotools scikit-image tqdm easydict \
        && pip install --ignore-installed ${tensorflow_ver} \
        && pip uninstall -y h5py \
        && pip uninstall -y h5py  \
        && mamba install -y --override-channels --force-reinstall h5py=2.10.0 -c conda-forge -c anaconda \
    && pip install --force --no-binary protobuf protobuf==3.19.4 \
    && conda clean -y --force-pkgs-dirs \
    && sudo rm -fr ~/.cache \
    && sudo rm -fr /scratch/* \
    && conda config --env --remove-key channels \
    && conda activate vitis-ai-tensorflow2 \
    && sudo mkdir -p $VAI_ROOT/compiler \
    && sudo cp -r $CONDA_PREFIX/lib/python3.7/site-packages/vaic/arch $VAI_ROOT/compiler/arch
fi
