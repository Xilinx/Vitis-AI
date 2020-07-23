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
Vitis AI Docker images is based on Ubuntu Bonic 18.04. The software packages sources */etc/apt/sources.list* looks like this:

```   
  deb http://us.archive.ubuntu.com/ubuntu/ bionic universe   
```
 

These hostname "archive.ubuntu.com" by default resolve to servers in the United States. Any customers building the Vitis AI GPU image  https://github.com/Xilinx/Vitis-AI/blob/master/docker/DockerfileGPU  will pull from these servers. Due to the Great Firewall of China, connections are very slow.

 

### Workaround
Change the Ubuntu apt sources.list to use servers in China. For example, https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ command provided at http://jira.xilinx.com/browse/ML-2564?focusedCommentId=3930615&page=com.atlassian.jira.plugin.system.issuetabpanels:comment-tabpanel#comment-3930615  
