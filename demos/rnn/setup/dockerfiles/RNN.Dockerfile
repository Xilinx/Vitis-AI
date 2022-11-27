# docker build . --build-arg BASE_IMAGE=${BASE_IMAGE}  -t ${DOCKER_IMG_NAME}
ARG BASE_IMAGE=xilinx/vitis-ai-gpu:latest
FROM $BASE_IMAGE

ARG DATE_RNN
ENV DATE_RNN=$DATE_RNN
ARG GIT_HASH_RNN
ENV GIT_HASH_RNN=$GIT_HASH_RNN

ARG VAI_CONDA_CHANNEL="file:///scratch/conda-channel"
ENV VAI_CONDA_CHANNEL=$VAI_CONDA_CHANNEL
ARG CONDA_PACKAGE_PREFIX="https://www.xilinx.com/bin/public/openDownload?filename="

ADD --chown=vitis-ai-user:vitis-ai-group dockerfiles/gpu_conda/* /scratch/
ADD --chown=vitis-ai-user:vitis-ai-group dockerfiles/banner_rnn.sh /scratch/
RUN cat /scratch/banner_rnn.sh >> /etc/banner.sh

RUN . $VAI_ROOT/conda/etc/profile.d/conda.sh \
    && mkdir -p /scratch/conda-channel/linux-64 \
    && chmod -R 777 /scratch \
    && cd /scratch/conda-channel/linux-64 \
    && wget --progress=dot:mega -O vart-1.4.1-py36h2333f42_130.tar.bz2 ${CONDA_PACKAGE_PREFIX}vart-1.4.1-py36h2333f42_130.tar.bz2 \
    && wget --progress=dot:mega -O vart-1.4.1-py37h2333f42_130.tar.bz2 ${CONDA_PACKAGE_PREFIX}vart-1.4.1-py37h2333f42_130.tar.bz2 \
    && wget --progress=dot:mega -O xir-1.4.1-py36h20355c0_91.tar.bz2 ${CONDA_PACKAGE_PREFIX}xir-1.4.1-py36h20355c0_91.tar.bz2 \
    && wget --progress=dot:mega -O xir-1.4.1-py37h20355c0_91.tar.bz2 ${CONDA_PACKAGE_PREFIX}xir-1.4.1-py37h20355c0_91.tar.bz2 \
    && wget --progress=dot:mega -O target_factory-1.4.1-h2914a11_84.tar.bz2 ${CONDA_PACKAGE_PREFIX}target_factory-1.4.1-h2914a11_84.tar.bz2 \
    && wget --progress=dot:mega -O unilog-1.4.1-h20355c0_82.tar.bz2 ${CONDA_PACKAGE_PREFIX}unilog-1.4.1-h20355c0_82.tar.bz2 \
    && wget --progress=dot:mega -O vai_c_rnn-1.4.1-py36h32e1ea0_5.tar.bz2 ${CONDA_PACKAGE_PREFIX}vai_c_rnn-1.4.1-py36h32e1ea0_5.tar.bz2 \
    && wget --progress=dot:mega -O tf_nndct_lstm-1.4.1-py36h7564e9b_32.tar.bz2 ${CONDA_PACKAGE_PREFIX}tf_nndct_lstm-1.4.1-py36h7564e9b_32.tar.bz2 \
    && wget --progress=dot:mega -O pytorch_nndct_lstm-1.4.1-py36h7d579db_32.tar.bz2 ${CONDA_PACKAGE_PREFIX}pytorch_nndct_lstm-1.4.1-py36h7d579db_32.tar.bz2 \
    && conda index /scratch/conda-channel \
    && cat /scratch/conda-channel/linux-64/repodata.json \
    && conda install mamba -n base -c conda-forge \
    && echo "channels:" >> /etc/conda/condarc \
    && echo "  - ${VAI_CONDA_CHANNEL}" >> /etc/conda/condarc \
    && echo "INFO: /etc/conda/condarc is \n $(cat /etc/conda/condarc)" \
    && mamba env create -f /scratch/rnn-tf-2.0.yml \
    && conda install -n rnn-tf-2.0 \
        /scratch/conda-channel/linux-64/vart-1.4.1-py36h2333f42_130.tar.bz2 \
        /scratch/conda-channel/linux-64/xir-1.4.1-py36h20355c0_91.tar.bz2 \
        /scratch/conda-channel/linux-64/target_factory-1.4.1-h2914a11_84.tar.bz2 \
     && conda activate rnn-tf-2.0

RUN . $VAI_ROOT/conda/etc/profile.d/conda.sh \
    && mamba env create -f /scratch/rnn-pytorch-1.4.yml \
    && conda install -n rnn-pytorch-1.4 \
        /scratch/conda-channel/linux-64/vart-1.4.1-py36h2333f42_130.tar.bz2 \
        /scratch/conda-channel/linux-64/xir-1.4.1-py36h20355c0_91.tar.bz2 \
        /scratch/conda-channel/linux-64/target_factory-1.4.1-h2914a11_84.tar.bz2 \
    && conda activate rnn-pytorch-1.4 \
    && pip install -r /scratch/pip_requirements_pytorch.txt

RUN . $VAI_ROOT/conda/etc/profile.d/conda.sh \
    && conda activate rnn-pytorch-1.4 \
    && python -m spacy download en_core_web_sm \
    && python -m nltk.downloader -d /usr/local/share/nltk_data all

RUN . $VAI_ROOT/conda/etc/profile.d/conda.sh \
    && mamba env create -f /scratch/vitis-ai-rnn.yml

RUN apt-get update -y && apt install -y sox \
    && cat /dev/null > /etc/conda/condarc

# Install VCK5000 Shells
RUN cd /tmp \
    && wget --progress=dot:mega -O xilinx-sc-fw-vck5000.deb https://www.xilinx.com/bin/public/openDownload?filename=xilinx-sc-fw-vck5000_4.4.8-1.fe3928b_all.deb \
    && wget --progress=dot:mega -O xilinx-vck5000-es1-gen3x16-base.deb https://www.xilinx.com/bin/public/openDownload?filename=xilinx-vck5000-es1-gen3x16-base_2-3123623_all.deb \
    && wget --progress=dot:mega -O xilinx-vck5000-es1-gen3x16-validate.deb https://www.xilinx.com/bin/public/openDownload?filename=xilinx-vck5000-es1-gen3x16-validate_2-3123623_all.deb \
    && apt install -y ./xilinx-sc-fw-vck5000.deb ./xilinx-vck5000-es1-gen3x16-base.deb ./xilinx-vck5000-es1-gen3x16-validate.deb \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/*

RUN . $VAI_ROOT/conda/etc/profile.d/conda.sh \
    && conda activate rnn-pytorch-1.4

RUN echo $DATE_RNN > /etc/BUILD_DATE.txt \
    && echo $GIT_HASH_RNN > /etc/GIT_HASH.txt \
    && sed -i "s/export\ BUILD_DATE.*/export\ BUILD_DATE=\"${DATE_RNN}\"/g" ~vitis-ai-user/.bashrc \
    && sed -i "s/export\ BUILD_DATE.*/export\ BUILD_DATE=\"${DATE_RNN}\"/g" /root/.bashrc \
    && sed -i "s/export\ GIT_HASH.*/export\ GIT_HASH=${GIT_HASH_RNN}/g" ~vitis-ai-user/.bashrc \
    && sed -i "s/export\ GIT_HASH.*/export\ GIT_HASH=${GIT_HASH_RNN}/g" /root/.bashrc \
    && sed -i "/export\ BUILD_DATE/ a export GIT_HASH=`cat /etc/GIT_HASH.txt`" /etc/bash.bashrc
