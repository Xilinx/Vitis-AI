# Installing Docker
Please refer to official Docker Documentation

- https://docs.docker.com/engine/install/

# Installing NVIDIA Docker Runtime
(Only applicable for model training or model quantization use cases)  
If you are building the Vitis AI Docker Image with GPU acceleration
You will need to install NVIDIA Container Runtime
Please refer to the offical NVIDIA Documentation

- https://nvidia.github.io/nvidia-container-runtime/


# Slow Connection To Ubuntu Servers Outside China
Vitis AI Docker images is based on Ubuntu Bonic 18.04. The software packages sources **/etc/apt/sources.list** looks like this:

```   
  deb http://us.archive.ubuntu.com/ubuntu/ bionic universe   
```
 
These hostname "archive.ubuntu.com" by default resolve to servers in the United States. Any customers building the Vitis AI GPU image https://github.com/Xilinx/Vitis-AI/blob/master/docker/DockerfileGPU will pull from these servers. Users accessing from China, might experience slow download speeds using these servers. 

### Workaround
Change the Ubuntu apt sources.list to use servers in China. 

In Dockerfile, change the 1st instance of:
```
  RUN apt-get update -y && apt-get install -y --no-install-recommends \
```

To the line below, so that the Ubuntu sources will be replaced with China Ubuntu servers
```      
  RUN sed --in-place --regexp-extended "s/(\/\/)(archive\.ubuntu)/\1cn.\2/" /etc/apt/sources.list && apt-get update && apt-get install -y --no-install-recommends \
```

The following Tsinghua University (Beijing) sources can also be used:

deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse

Other alternate sources:

https://momane.com/change-ubuntu-18-04-source-to-china-mirror

