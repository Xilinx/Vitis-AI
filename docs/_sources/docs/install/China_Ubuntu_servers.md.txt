<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>


## Access to Ubuntu mirrors from within China
Vitis AI Docker images leverage Ubuntu Bionic 18.04. In your Ubuntu installation, the file **/etc/apt/sources.list** specifies the default server location for Ubuntu packages.  For example:

```   
  deb http://us.archive.ubuntu.com/ubuntu/ bionic universe   
```

You can see that the hostname "archive.ubuntu.com" resolves to servers located within the United States. When building the Vitis AI Docker image, whether for [CPU-only](https://github.com/Xilinx/Vitis-AI/blob/master/docker/dockerfiles/vitis-ai-cpu.Dockerfile) or [GPU](https://github.com/Xilinx/Vitis-AI/blob/master/docker/dockerfiles/vitis-ai-gpu.Dockerfile) applications Docker will attempt to pull from US servers. Users accessing from China will generally experience slow download speeds as a result.

Prior to building the Vitis AI Docker image it is recommended that you modify **/etc/apt/sources.list** as well as the vitis-ai-gpu.Dockerfile.

In the Vitis AI .DockerFile, change the first instances of apt-get update and apt-get install.

From:

```
  RUN chmod 1777 /tmp \
    .......
    .......
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
  ```

To:
```      
  RUN sed --in-place --regexp-extended "s/(\/\/)(archive\.ubuntu)/\1cn.\2/" /etc/apt/sources.list && apt-get update && apt-get install -y --no-install-recommends \
```

Next, modify your Ubuntu **/etc/apt/sources.list** to point to mirrors located in China.  The following Tsinghua University (Beijing) mirrors are recommended:

****************
:pushpin:**Important Note!** Please backup **/etc/apt/sources.list** prior to making this change 
****************

```
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
```

In addition, the below URL provides multiple alternative mirrors:

https://momane.com/change-ubuntu-18-04-source-to-china-mirror
