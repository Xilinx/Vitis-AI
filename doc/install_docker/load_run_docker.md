**PATH_TO is the directory where the downloaded docker containers are located in your Linux machine.**  

**<x.y.z> is the version of the docker containers. Please replace it with the actual version of the downloaded docker containers.**

**<DOCK_NAME> is dock name which is displayed after loading the docker succesfully. For example, xdock.xilinx.com/vitis-ai:<x.y.z>-gpu.  
Please replace it with the actual name of the loaded docker.**

**The GPU docker has been tested with GPU machines with Docker 19.03.1, NVIDIA driver 410.xx (to work with CUDA 10.0) and nvidia-docker 2.2.2.**

## CPU tools docker

```shell
docker load < /PATH_TO/vitis-ai-tools-<x.y.z>-cpu.tar.gz
./docker_run.sh xdock:5000/vitis-ai:<x.y.z>-cpu
```

## GPU tools docker

```shell
docker load < /PATH_TO/vitis-ai-tools-<x.y.z>-gpu.tar.gz
./docker_run.sh xdock:5000/vitis-ai:<x.y.z>-gpu
```

## Runtime docker

```shell
docker load < /PATH_TO/vitis-ai-runtime-<x.y.z>.tar.gz
./docker_run.sh xdock.xilinx.com/vitis-ai-runtime:<x.y.z>
```
