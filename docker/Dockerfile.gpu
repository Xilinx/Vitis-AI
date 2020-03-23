#################################################################################

#NOTICE: CAREFULLY READ THIS DOCKERFILE.  BY INVOKING THIS FILE IN DOCKER AND USING THE SOFTWARE INSTALLED BY DOCKER, YOU AGREE ON BEHALF OF YOURSELF AND YOUR EMPLOYER (IF APPLICABLE) (INDIVIDUALLY AND COLLECTIVELY, “LICENSEE”) TO BE BOUND BY THIS INFORMATION AND THE LICENSE AGREEMENTS APPLICABLE TO THE SPECIFIC SOFTWARE INSTALLED IN THE DOCKER CONTAINER.  IF LICENSEE DOES NOT AGREE TO ALL OF THE INFORMATION AND APPLICABLE LICENSE AGREEMENTS, DO NOT INVOKE THIS FILE IN DOCKER.

#THIS DOCKERFILE AUTOMATICALLY INSTALLS A VARIETY OF SOFTWARE COPYRIGHTED BY XILINX AND THIRD PARTIES (INDIVIDUALLY AND COLLECTIVELY, “SOFTWARE”) THAT IS SUBJECT TO VARIOUS “OPEN SOURCE” OR “PROPRIETARY” LICENSE AGREEMENTS (INDIVIDUALLY AND COLLECTIVELY, “LICENSE AGREEMENTS”) THAT APPEAR UPON INSTALLATION, ACCEPTANCE AND/OR ACTIVATION OF THE SOFTWARE AND/OR ARE CONTAINED OR DESCRIBED IN THE CORRESPONDING RELEASE NOTES OR OTHER DOCUMENTATION OR HEADER OR SOURCE FILES.  XILINX DOES NOT GRANT TO LICENSEE ANY RIGHTS OR LICENSES TO SUCH THIRD-PARTY SOFTWARE.  LICENSEE AGREES TO CAREFULLY REVIEW AND ABIDE BY THE TERMS AND CONDITIONS OF SUCH LICENSE AGREEMENTS TO THE EXTENT THAT THEY GOVERN SUCH SOFTWARE.

#################################################################################


FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04
env DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV VAI_ROOT=/opt/vitis_ai
ARG VERSION
ENV VERSION=$VERSION
ENV PYTHONPATH=/opt/vitis_ai/compiler
ARG DATE
ENV DATE="$DATE"

#################################################################################

RUN apt-get update && apt-get install -y --no-install-recommends \
	sudo \
	git \
	zstd \
	tree \
	vim \
	wget \ 
	bzip2 \
	ca-certificates \
	curl \
	unzip \
	python3-minimal \
	python3-opencv \
	python3-venv \
	python3-pip \
	python3-setuptools \
	g++ \
	make \
	cmake \
	build-essential \
	autoconf \
	libgoogle-glog-dev \
	libgflags-dev \
	libunwind-dev \
	libtool

#################################################################################

RUN pip3 install \
	Flask \
	setuptools \
	wheel

#################################################################################

RUN mkdir -p $VAI_ROOT/compression && \
	ln -s $VAI_ROOT/conda/envs/vitis-ai-caffe/bin/decent_q $VAI_ROOT/compression/vai_q_caffe && \
	ln -s $VAI_ROOT/conda/envs/vitis-ai-tensorflow/bin/decent_q $VAI_ROOT/compression/vai_q_tensorflow

RUN mkdir -p $VAI_ROOT/compiler && \
	ln -s $VAI_ROOT/conda/envs/vitis-ai-caffe/lib/python3.6/site-packages/vai/dpuv1/tools/compile/bin/vai_c_caffe.py $VAI_ROOT/compiler/vai_c_caffe && \
	ln -s $VAI_ROOT/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/vai/dpuv1/tools/compile/bin/vai_c_tensorflow.py $VAI_ROOT/compiler/vai_c_tensorflow

#################################################################################

RUN cd $VAI_ROOT/compiler && wget https://www.xilinx.com/bin/public/openDownload?filename=compiler.tar.gz -O compiler.tar.gz && tar -xzvf compiler.tar.gz && mv ./compiler/* ./ && rm -rf compiler.tar.gz ./compiler
RUN cd $VAI_ROOT && wget https://www.xilinx.com/bin/public/openDownload?filename=utility.tar.gz -O utility.tar.gz && tar -xzvf utility.tar.gz && rm -rf utility.tar.gz

ENV PATH="${VAI_ROOT}/utility:${PATH}"

#################################################################################

RUN mkdir -p /scratch && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /scratch/miniconda.sh && \
	/bin/bash /scratch/miniconda.sh -b -p $VAI_ROOT/conda && \
	ln -s $VAI_ROOT/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
	echo ". $VAI_ROOT/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
	echo "conda activate base" >> ~/.bashrc

#################################################################################

ADD ./conda_requirements.txt /scratch/
ADD ./conda_requirements_neptune.txt /scratch/
ADD ./pip_requirements.txt /scratch/
ADD ./pip_requirements_neptune.txt /scratch/

RUN cd /scratch && wget https://www.xilinx.com/bin/public/openDownload?filename=conda-channel.tar.gz -O conda-channel.tar.gz && tar -xzvf conda-channel.tar.gz

# Create Caffe && TensorFlow conda envs
RUN . $VAI_ROOT/conda/etc/profile.d/conda.sh && \
	conda create -n vitis-ai-caffe \
	python=3.6 caffe_decent_gpu \
	--file /scratch/conda_requirements.txt \
	-c file:///scratch/conda-channel -c defaults -c conda-forge/label/gcc7 && \
	conda activate vitis-ai-caffe && pip install -r /scratch/pip_requirements.txt && \
	conda create -n vitis-ai-tensorflow \
	python=3.6 tensorflow_decent_gpu \
	--file /scratch/conda_requirements.txt \
	-c file:///scratch/conda-channel -c defaults -c conda-forge/label/gcc7 && \
	conda activate vitis-ai-tensorflow && pip install -r /scratch/pip_requirements.txt && \
	conda create -n vitis-ai-neptune \
	python=3.6 \
	--file /scratch/conda_requirements_neptune.txt \
	-c file:///scratch/conda-channel -c defaults -c conda-forge/label/gcc7 -c conda-forge && \
	conda activate vitis-ai-neptune && pip install -r /scratch/pip_requirements_neptune.txt

#################################################################################

RUN wget https://www.xilinx.com/bin/public/openDownload?filename=xrt_201920.2.3.1301_18.04-xrt.deb -O scratch/xrt.deb && \
	apt-get install -y --no-install-recommends /scratch/*.deb 

#################################################################################

RUN apt-get clean && rm -rf /var/lib/apt/lists/* && rm -rf /scratch/*

ADD ./login.sh /etc/
ENTRYPOINT ["/etc/login.sh"]

ADD ./banner.sh /etc/

#################################################################################

