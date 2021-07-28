# docker build . --build-arg BASE_IMAGE=${BASE_IMAGE}  -t ${DOCKER_IMG_NAME}
ARG BASE_IMAGE=xilinx/vitis-ai:latest
FROM $BASE_IMAGE
RUN . $VAI_ROOT/conda/etc/profile.d/conda.sh \
    && conda create -n rnn_tf_2.0 \
        python=3.7 Pandas pybind11 \
        -c defaults -c conda-forge \
    && conda activate rnn_tf_2.0 \
    && pip install \
        keras==2.3.1 \
        tensorflow==2.0.0 \
        overrides \
        tqdm \
        docopt \
        allennlp \
        https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz \
    && conda create -n rnn_pytorch_1.4 \
        python=3.6.10  Pandas Keras pybind11 \
        -c defaults -c conda-forge \
    && conda activate rnn_pytorch_1.4 \
    && pip install \
        ftfy pytorch_pretrained_bert \
        word2number \
        conllu \
        overrides==3.0.0 \
        spaCy==2.1 \
        tqdm==4.48.2 \
        docopt \
        allennlp \
        pytorch-transformers \
        https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz \
        toml \
        torch==1.4.0 \
        torchvision==0.5.0 \
        metrics \
        numpy==1.19.1 \
        librosa \
        SoundFile \
        prettytable \
        Unidecode==1.1.1 \
        inflect==4.1.0 \
        sox \
    && apt-get update -y && apt install -y sox \
    && echo "echo -e 'For xrnn TensorFlow Workflows do:\n  conda activate rnn_tf_2.0\nFor xrnn Pytorch Workflows do:\n  conda activate rnn_pytorch_1.4'" >> /etc/banner.sh 

# Install VCK5000 Shells
RUN cd /tmp \
    && wget --progress=dot:mega -O xilinx-sc-fw-vck5000.deb http://xcogpuvai01.xilinx.com:8000/xilinx-sc-fw-vck5000_4.4.8-1.fe3928b_all.deb \
    && wget --progress=dot:mega -O xilinx-vck5000-es1-gen3x16-base.deb http://xcogpuvai01.xilinx.com:8000/xilinx-vck5000-es1-gen3x16-base_2-3123623_all.deb \
    && wget --progress=dot:mega -O xilinx-vck5000-es1-gen3x16-validate.deb http://xcogpuvai01.xilinx.com:8000/xilinx-vck5000-es1-gen3x16-validate_2-3123623_all.deb \
    && apt install -y ./xilinx-sc-fw-vck5000.deb ./xilinx-vck5000-es1-gen3x16-base.deb ./xilinx-vck5000-es1-gen3x16-validate.deb \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/*

RUN . $VAI_ROOT/conda/etc/profile.d/conda.sh \
    && conda activate rnn_pytorch_1.4 \
    && python -m nltk.downloader -d /usr/local/share/nltk_data all

