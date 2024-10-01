#!/bin/bash                                                                                                                                                                     
  
set -ex
sudo chmod 777 /scratch
if [[ ${VAI_CONDA_CHANNEL} =~ .*"tar.gz" ]]; then \
       cd /scratch/; \
       wget -O conda-channel.tar.gz --progress=dot:mega ${VAI_CONDA_CHANNEL}; \
       tar -xzvf conda-channel.tar.gz; \
       export VAI_CONDA_CHANNEL=file:///scratch/conda-channel; \
fi;
sudo mkdir -p $VAI_ROOT/compiler 

if [[ ${DOCKER_TYPE} != 'cpu' ]]; then \
    arch_type="_${DOCKER_TYPE}";
else
    arch_type="";

fi

conda_channel="${VAI_CONDA_CHANNEL}"

if [[ ${DOCKER_TYPE} == 'rocm' ]]; then \
    tensorflow_ver="tensorflow-${DOCKER_TYPE}==2.11.1.550  keras==2.11";
#    conda_channel="${VAI_CONDA_CHANNEL}"
else
    tensorflow_ver="tensorflow==2.12 keras==2.12";

fi
if [[ ${DOCKER_TYPE} == 'cpu' ]]; then
   . $VAI_ROOT/conda/etc/profile.d/conda.sh \
    && mkdir -p $VAI_ROOT/conda/pkgs \
    && python3 -m pip install --upgrade pip wheel setuptools \
    && conda config --env --append channels ${VAI_CONDA_CHANNEL} \
    && conda config --remove channels defaults || true \
    && cat ~/.condarc \
    && mamba env create -f /scratch/${DOCKER_TYPE}_conda/vitis-ai-tensorflow2.yml \
    && conda activate vitis-ai-tensorflow2 \
    && mamba install --no-update-deps  vai_q_tensorflow2 pydot pyyaml jupyter ipywidgets \
            dill progressbar2 pytest pandas matplotlib \
             -c ${VAI_CONDA_CHANNEL} -c conda-forge \
        && pip install -r /scratch/pip_requirements.txt \
        && pip install transformers protobuf==3.20.3 pycocotools scikit-learn scikit-image tqdm easydict onnx==1.13.0 numpy==1.22 \
        &&  pip install --force-reinstall wrapt==1.14 absl-py astunparse gast google-pasta grpcio jax keras==2.12  libclang opt-einsum tensorboard tensorflow-estimator==2.12  termcolor \
        && pip uninstall -y h5py \
        && pip uninstall -y h5py \
        && mamba install -y --override-channels --force-reinstall h5py=2.10.0 tensorflow-onnx zendnn-tensorflow2  -c conda-forge \
        && pip install --force-reinstall numpy==1.22 protobuf==3.20.3 \
    && conda config --env --remove-key channels \
    && conda clean -y --force-pkgs-dirs \
    && sudo cp -r $CONDA_PREFIX/lib/python3.8/site-packages/vaic/arch $VAI_ROOT/compiler/arch \
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
    && mamba install /scratch/conda-channel/linux-64/tensorflow-onnx-3.5.0-hcdf1d9b_18.tar.bz2 \
    && mamba install --no-update-deps -y  pydot pyyaml jupyter ipywidgets \
            dill progressbar2 pytest scikit-learn pandas matplotlib \
            pillow -c ${conda_channel} -c conda-forge -c defaults \
        && pip install -r /scratch/pip_requirements.txt \
        && pip install pycocotools scikit-image tqdm easydict \
        && pip install --ignore-installed tensorflow-rocm==2.11.1.550  keras==2.11 \
        && pip install --force --no-binary protobuf protobuf==3.20.3 \
        && pip uninstall -y h5py \
        && pip uninstall -y h5py  \
        && mamba install -y --override-channels --force-reinstall h5py=2.10.0 -c conda-forge  \
    && conda clean -y --force-pkgs-dirs \
    && sudo rm -fr ~/.cache \
    && sudo rm -fr /scratch/* \
    && conda config --env --remove-key channels \
    && conda activate vitis-ai-tensorflow2 \
    && sudo mkdir -p $VAI_ROOT/compiler \
    && sudo cp -r $CONDA_PREFIX/lib/python3.8/site-packages/vaic/arch $VAI_ROOT/compiler/arch
else
. $VAI_ROOT/conda/etc/profile.d/conda.sh \
    && mkdir -p $VAI_ROOT/conda/pkgs \
    && sudo python3 -m pip install --upgrade pip wheel setuptools \
    && conda config --env --remove-key channels \
    && conda config --env --append channels ${VAI_CONDA_CHANNEL} \
    && conda config --remove channels defaults || true \
    && mamba env create -f /scratch/${DOCKER_TYPE}_conda/vitis-ai-tensorflow2.yml \
    && conda activate vitis-ai-tensorflow2 \
    && pip install --ignore-installed ${tensorflow_ver} \
    && mamba install --no-update-deps -y  pydot pyyaml jupyter ipywidgets \
            dill progressbar2 pytest pandas matplotlib \
            pillow -c ${conda_channel} -c conda-forge -c defaults \
        && pip install -r /scratch/pip_requirements.txt \
        && pip install transformers pycocotools scikit-learn scikit-image tqdm easydict \
        && pip install --ignore-installed ${tensorflow_ver} \
        && pip uninstall -y h5py \
        && pip uninstall -y h5py  \
        && mamba install -y --override-channels --force-reinstall h5py=2.10.0 -c conda-forge  \
    && pip install --force --no-binary protobuf protobuf==3.20.3 \
    && conda clean -y --force-pkgs-dirs \
    && sudo rm -fr ~/.cache \
    && sudo rm -fr /scratch/* \
    && conda config --env --remove-key channels \
    && conda activate vitis-ai-tensorflow2 \
    && sudo mkdir -p $VAI_ROOT/compiler \
    && sudo cp -r $CONDA_PREFIX/lib/python3.8/site-packages/vaic/arch $VAI_ROOT/compiler/arch
fi
