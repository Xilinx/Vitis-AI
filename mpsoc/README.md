This directory contains instructions and examples for running DPU-v2 on Zynq Ultrascale+ MPSoC platforms. It can also be applied to Zynq-7000 platforms.
**DPU-v2**  is a configurable computation engine dedicated for convolutional neural networks. 
It includes a set of highly optimized instructions, and supports most convolutional neural networks, such as VGG, ResNet, GoogleNet, YOLO, SSD, MobileNet, FPN, and others.
With Vitis-AI, Xilinx has integrated all the edge and cloud solutions under a unified API and toolset.

**Learn More:** [DPU-v2 Overview](https://github.com/Xilinx/Vitis-AI/tree/master/DPU-TRD)  

## Step1: Install Docker

Install the Docker on the host, and add the user to the docker group. Link the user to docker installation instructions from the following docker's website:

- [https://docs.docker.com/install/linux/docker-ce/ubuntu/](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
- [https://docs.docker.com/install/linux/docker-ce/centos/](https://docs.docker.com/install/linux/docker-ce/centos/)
- [https://docs.docker.com/install/linux/linux-postinstall/](https://docs.docker.com/install/linux/linux-postinstall/)

## Step2: Setup Docker Image

1. Clone the Vitis AI repository
```sh
git clone https://github.com/xilinx/vitis-ai
```
2. Start the Docker Container without GPU card
```sh
cd Vitis-AI/
./docker_run.sh xilinx/vitis-ai:1.0.0-cpu
```

3. Or start the Docker Container with GPU card
```sh
cd Vitis-AI/docker
./docker_build.sh
cd Vitis-AI
./docker_run.sh xilinx/vitis-ai:1.0.0-gpu
```

4. Upon starting the container your current working directory will be mounted to: /
workspace

## Step3: Setup cross-compiler for Vitis AI DNNDK and make samples

1. Download [sdk.sh](https://www.xilinx.com/bin/public/openDownload?filename=sdk.sh)

2. Run the command below to install Arm GCC cross-compilation toolchain environment
```sh
./sdk.sh
```

3. Run the command below to setup environment
```sh
source /opt/petalinux/2019.2/environment-setup-aarch64-xilinx-linux
```

4. Download DNNDK runtime package [vitis-ai_v1.1_dnndk.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai_v1.1_dnndk.tar.gz) and install it into PetaLinux system sysroot
```sh
tar -xzvf vitis-ai_v1.1_dnndk.tar.gz
cd vitis-ai_v1.1_dnndk
sudo ./install.sh $SDKTARGETSYSROOT
```

5. Cross compile samples
```sh
cd Vitis-AI/mpsoc/vitis_ai_dnndk_samples/resnet50
# For ZCU102 evaluation board,
./build.sh zcu102
# or for ZCU104 evaluation board
./build.sh zcu104
```

## Step4: Setup Evaluation Board and run Vitis AI DNNDK samples

Download the prebuilt [ZCU102 board image](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-zcu102-dpu-v2019.2-v2.img.gz) or [ZCU104 board image](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-zcu104-dpu-v2019.2-v2.img.gz) and flash it to SD card (16GB recommanded) using Etcher or Win32DiskImager. Note that you may need to run command irps5401 first to trigger the power management patch for ZCU104 to avoid system hang or power off issue when running samples.

1. Copy the package [vitis-ai_v1.1_dnndk.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai_v1.1_dnndk.tar.gz) to /home/root directory of the evaluation board, and follow the steps below to set up the environment on the board
```sh
tar -xzvf vitis-ai_v1.1_dnndk.tar.gz
cd vitis-ai_v1.1_dnndk
./install.sh
```

2. Copy the Vitis-AI/mpsoc/vitis_ai_dnndk_samples folder from host to the /home/root directory of the evaluation board

3. Download [vitis-ai_v1.1_dnndk_sample_img.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai_v1.1_dnndk_sample_img.tar.gz) package to /home/root of the evaluation board and run command below
```sh
tar -xzvf vitis-ai_v1.1_dnndk_sample_img.tar.gz
```

4. Run the samples on evaluation board
```sh
cd /home/root/vitis-ai_v1.1_dnndk_sample/resnet50
./resnet50

```

## References 
- [Use Tool Docker](tool_docker.md)
- [Vitis AI User Guide](https://www.xilinx.com/cgi-bin/docs/ndoc?t=user_guides;d=ug1414-vitis-ai.pdf)
- [Vitis AI Model Zoo](https://github.com/Xilinx/Vitis-AI/tree/master/AI-Model-Zoo)

