**Internet access is required.**

## Reference: https://docs.docker.com/install/linux/docker-ce/ubuntu/

The following instructions have been tested on Ubuntu 18.04.

## Add the Docker repository into your Ubuntu host

```shell
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"  
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -  
sudo apt-get update && sudo apt install docker-ce docker-ce-cli containerd.io 
```

## Install NVIDIA Docker Runtime

Instructions from https://nvidia.github.io/nvidia-container-runtime/

```shell
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \  
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
sudo apt-get install nvidia-container-runtime
```


The edit the docker config to limit / allow users in docker group membership to run Docker containers

```shell
sudo systemctl edit docker
```

This is the content
```shell
[Service]
ExecStart=
ExecStart=/usr/bin/dockerd --group docker -H unix:///var/run/docker.sock --add-runtime=nvidia=/usr/bin/nvidia-container-runtime
```

Then restart the InitD daemon and restart Docker
```shell
sudo systemctl daemon-reload
sudo systemctl restart docker
```


