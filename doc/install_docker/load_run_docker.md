**Docker Images Are Hosted on Docker Hub [HERE](https://hub.docker.com/repository/docker/xilinx/vitis-ai)**  

**The GPU docker has been tested with GPU machines with Docker 19.03.1, NVIDIA driver 410.xx (to work with CUDA 10.0) and nvidia-docker 2.2.2.**

## CPU tools docker

```shell
# Load the CPU image from dockerhub

./docker_run.sh xilinx/vitis-ai-cpu:latest

Or 

# Build the CPU image and load it

cd Vitis-AI/docker
./docker_build_cpu.sh
cd ..
./docker_run.sh xilinx/vitis-ai-cpu:latest

```

## GPU tools docker

```shell
# GPU Image must be built by user. 

cd Vitis-AI/docker
./docker_build_gpu.sh

# After the GPU image is built, load it

cd ..
./docker_run.sh xilinx/vitis-ai-gpu:latest

```

