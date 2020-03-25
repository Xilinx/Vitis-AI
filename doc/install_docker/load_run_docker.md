**Docker Images Are Hosted on Docker Hub [HERE](https://hub.docker.com/repository/docker/xilinx/vitis-ai)**  

**<x.y.z> is the version of the docker containers. Please replace it with the actual version of the downloaded docker containers.**

**The GPU docker has been tested with GPU machines with Docker 19.03.1, NVIDIA driver 410.xx (to work with CUDA 10.0) and nvidia-docker 2.2.2.**

## CPU tools docker

```shell

cd Vitis-AI/docker
./docker_build_cpu.sh

# After the CPU image is built, load it

cd Vitis-AI
./docker_run.sh xilinx/vitis-ai-cpu:1.1.48

```

## GPU tools docker

```shell
# GPU Image must be built by user. 

cd Vitis-AI/docker
./docker_build_gpu.sh

# After the GPU image is built, load it

cd Vitis-AI
./docker_run.sh xilinx/vitis-ai-gpu:1.1.48

```

