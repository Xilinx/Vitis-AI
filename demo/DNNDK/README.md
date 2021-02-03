This directory contains instructions and examples for running DPUCZDX8G on Zynq Ultrascale+ MPSoC platforms. It can also be applied to Zynq-7000 platforms.
**DPUCZDX8G**  is a configurable computation engine dedicated for convolutional neural networks. 
It includes a set of highly optimized instructions, and supports most convolutional neural networks, such as VGG, ResNet, GoogleNet, YOLO, SSD, MobileNet, FPN, and others.
With Vitis-AI, Xilinx has integrated all the edge and cloud solutions under a unified API and toolset.

**Learn More:** [DPUCZDX8G Overview](../../dsa/DPU-TRD)  


Assume you have run the docker and the current working directory is /workspace

## Step1: Setup cross-compiler for Vitis AI DNNDK and make samples

1. Download [sdk-2020.2.0.0.sh](https://www.xilinx.com/bin/public/openDownload?filename=sdk-2020.2.0.0.sh)

2. Run the command below to install Arm GCC cross-compilation toolchain environment
```sh
./sdk-2020.2.0.0.sh
```

3. Run the command below to setup environment
```sh
source /opt/petalinux/2020.2/environment-setup-aarch64-xilinx-linux
```

4. Cross compile samples
```sh
cd Vitis-AI/demo/DNNDK/resnet50
# For ZCU102 evaluation board,
./build.sh zcu102
# or for ZCU104 evaluation board
./build.sh zcu104
```

## Step2: Setup Evaluation Board and run Vitis AI DNNDK samples

Download the prebuilt [ZCU102 board image](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-zcu102-dpu-v2020.2-v1.3.0.img.gz) or [ZCU104 board image](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-zcu104-dpu-v2020.2-v1.3.0.img.gz) and flash it to SD card (16GB recommanded) using Etcher or Win32DiskImager. Note that you may need to run command irps5401 first to trigger the power management patch for ZCU104 to avoid system hang or power off issue when running samples. Follow the steps below to run the samples.

1. Copy the package [vitis-ai_v1.3_dnndk.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai_v1.3_dnndk.tar.gz) to /home/root directory of the evaluation board. If you want to update the Vitis AI DNNDK Runtime or install them to your custom board image, follow these steps.
```sh
tar -xzvf vitis-ai_v1.3_dnndk.tar.gz
cd vitis-ai_v1.3_dnndk
./install.sh
```

2. Copy the Vitis-AI/demo/DNNDK folder from host to the /home/root/ directory of the evaluation board

3. Download [vitis-ai_v1.3_dnndk_sample_img.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai_v1.3_dnndk_sample_img.tar.gz) package to /home/root of the evaluation board and run command below
```sh
cd /home/root
tar -xzvf vitis-ai_v1.3_dnndk_sample_img.tar.gz
```

4. If you use your own system image, or there is no dpu.xclbin in the /usr/lib path. you may need to copy dpu.xclbin to /usr/lib first. Such as
```sh
cp /media/sd-mmcblk0p1/dpu.xclbin /usr/lib/
```

5. Run the samples on evaluation board
```sh
cd /home/root/DNNDK/resnet50
./resnet50

```
6. Run the other samples according to the following chapters of the Vitis AI User Guide, Quick Start -> Running Examples -> Legacy DNNDK Examples. For video input, only webm and raw format are supported by default with the above system image. If you want to support video data in other formats, you need to install the relevant packages on the system.
