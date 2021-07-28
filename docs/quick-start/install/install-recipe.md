<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1>
   </td>
 </tr>
</table>

# Build from Recipe

There are two types of docker recipes provided - CPU recipe and GPU recipe. If you have a compatible nVidia graphics card with CUDA support, you could use GPU recipe; otherwise you could use CPU recipe.

**CPU Docker**

Use below commands to build the CPU docker:
```
cd setup/docker
./docker_build_cpu.sh
```
To run the CPU docker, use command:
```
./docker_run.sh xilinx/vitis-ai-cpu:latest
```
**GPU Docker**

Use below commands to build the GPU docker:
```
cd setup/docker
./docker_build_gpu.sh
```
To run the GPU docker, use command:
```
./docker_run.sh xilinx/vitis-ai-gpu:latest
```
Please use the file **./docker_run.sh** as a reference for the docker launching scripts, you could make necessary modification to it according to your needs.
More Detail can be found here: [Run Docker Container](./install_docker/load_run_docker.md)

**X11 Support for Running Vitis AI Docker with Alveo**

If you are running Vitis AI docker with Alveo card and want to use X11 support for graphics (for example, some demo applications in VART and Vitis-AI-Library for Alveo need to display images or video), please add following line into the *docker_run_params* variable definition in *docker_run.sh* script:

~~~
-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/tmp/.Xauthority \
~~~

And after the docker starts up, run following command lines:

~~~
cp /tmp/.Xauthority ~/
sudo chown vitis-ai-user:vitis-ai-group ~/.Xauthority
~~~

Please note before running this script, please make sure either you have local X11 server running if you are using Windows based ssh terminal to connect to remote server, or you have run **xhost +** command at a command terminal if you are using Linux with Desktop. Also if you are using ssh to connect to the remote server, remember to enable *X11 Forwarding* option either with Windows ssh tools setting or with *-X* options in ssh command line.
