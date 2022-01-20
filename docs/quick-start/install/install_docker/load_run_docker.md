<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1>
   </td>
 </tr>
</table>

# Load and Run Docker

**Docker Images Are Hosted on Docker Hub [HERE](https://hub.docker.com/r/xilinx/vitis-ai/tags)**  

**The GPU docker has been tested with GPU machines with Docker 19.03.1, NVIDIA driver 410.xx (to work with CUDA 10.0) and nvidia-docker 2.2.2.**

## The tags of docker image on Docker Hub

There is a corresponding relationship between Vitis AI and docker image version, usually you need to find the docker image version corresponding to Vitis AI to use.
Among them, the easiest way is to directly use the pre-built docker image on Docker Hub. Just need to execute the following command. For the latest version of Vitis AI, <x.y.z> can simply be replaced with latest.
```shell
./docker_run.sh xilinx/vitis-ai-cpu:<x.y.z>
```

The version correspondence between Vitis AI and docker image is shown in the following table.

<table>
 <tr><th>Vitis AI Version</th><th>Docker image tag</th></tr>
 <tr><td>Vitis AI v2.0.0</td><td>./docker_run.sh xilinx/vitis-ai-cpu:2.0.0.1103</td></tr>
 <tr><td>Vitis AI v1.4.1</td><td>./docker_run.sh xilinx/vitis-ai-cpu:1.4.1.978</td></tr>
 <tr><td>Vitis AI v1.4</td><td>./docker_run.sh xilinx/vitis-ai-cpu:1.4.916</td></tr>
 <tr><td>Vitis AI v1.3</td><td rowspan="3">./docker_run.sh xilinx/vitis-ai-cpu:1.3.411</td></tr>
 <tr><td>Vitis AI v1.3.1</td></tr>
 <tr><td>Vitis AI v1.3.2</td></tr>
 <tr><td>Vitis AI v1.2</td><td rowspan="2">./docker_run.sh xilinx/vitis-ai-cpu:1.2.82</td></tr>
 <tr><td>Vitis AI v1.2.1</td></tr>
 <tr><td>Vitis AI v1.1</td><td rowspan="2">./docker_run.sh xilinx/vitis-ai-cpu:1.1.56</td></tr>
 <tr><td>Vitis AI v1.1-ultra96v2</td></tr>
 <tr><td rowspan="2">Vitis AI v1.0</td><td>docker pull xilinx/vitis-ai:tools-1.0.0-cpu</td></tr>
   <tr><td>docker pull xilinx/vitis-ai:runtime-1.0.0-cpu</td></tr>
</table>

## Using CPU tools docker

Simply, you only need to start the CPU docker image with the following command. If the specified docker image does not exist, the command will automatically download it from Docker Hub.

```shell
./docker_run.sh xilinx/vitis-ai-cpu:latest
```

If you don't want to download the image from Docker Hub, but build it from Docker file. Then you need to execute the command in the following way to build the docker image.

```shell
cd Vitis-AI/docker
./docker_build_cpu.sh
cd ..
```

Then you can use the same command as following to load and run the built docker image.

```shell
./docker_run.sh xilinx/vitis-ai-cpu:latest
```

## Using GPU tools docker

For GPU docker image, you must build it from the docker file, and there is no pre-built image. The build command is as follows.

```shell
cd Vitis-AI/docker
./docker_build_gpu.sh
cd ..
```

Then you can use the command as following to load and run the built docker image.

```shell
./docker_run.sh xilinx/vitis-ai-gpu:latest
```
