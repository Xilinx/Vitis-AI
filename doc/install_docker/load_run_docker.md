**Docker Images Are Hosted on Docker Hub [HERE](https://hub.docker.com/repository/docker/xilinx/vitis-ai)**  

**<x.y.z> is the version of the docker containers. Please replace it with the actual version of the downloaded docker containers.**

**The GPU docker has been tested with GPU machines with Docker 19.03.1, NVIDIA driver 410.xx (to work with CUDA 10.0) and nvidia-docker 2.2.2.**

## CPU tools docker

```shell
./docker_run.sh xilinx/vitis-ai:tools-<x.y.z>-cpu

# i.e.

./docker_run.sh xilinx/vitis-ai:tools-1.0.0-cpu
```

## GPU tools docker

```shell
# GPU Image must be built by user. Stay tuned for directions.
```

## Runtime docker

```shell
./docker_run.sh xilinx/vitis-ai:runtime-<x.y.z>-cpu

# i.e.

./docker_run.sh xilinx/vitis-ai:runtime-1.0.0-cpu
```
