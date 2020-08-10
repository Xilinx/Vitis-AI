This directory contains instructions and examples for running DPUCZDX8G on Zynq Ultrascale+ MPSoC platforms. It can also be applied to Zynq-7000 platforms.
**DPUCZDX8G**  is a configurable computation engine dedicated for convolutional neural networks. 
It includes a set of highly optimized instructions, and supports most convolutional neural networks, such as VGG, ResNet, GoogleNet, YOLO, SSD, MobileNet, FPN, and others.
With Vitis-AI, Xilinx has integrated all the edge and cloud solutions under a unified API and toolset.

**Learn More:** [DPUCZDX8G Overview](https://github.com/Xilinx/Vitis-AI/tree/master/DPU-TRD)  


Assume you have run the docker and the current working directory is /workspace

## Step1: Setup cross-compiler for Vitis AI DNNDK and make samples

1. Download [sdk-2020.1.0.0.sh](https://www.xilinx.com/bin/public/openDownload?filename=sdk-2020.1.0.0.sh)

2. Run the command below to install Arm GCC cross-compilation toolchain environment
```sh
./sdk-2020.1.0.0.sh
```

3. Run the command below to setup environment
```sh
source /opt/petalinux/2020.1/environment-setup-aarch64-xilinx-linux
```

4. Cross compile samples
```sh
cd Vitis-AI/mpsoc/vitis_ai_dnndk_samples/resnet50
# For ZCU102 evaluation board,
./build.sh zcu102
# or for ZCU104 evaluation board
./build.sh zcu104
```

## Step2: Setup Evaluation Board and run Vitis AI DNNDK samples

Download the prebuilt [ZCU102 board image](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-zcu102-dpu-v2020.1-v1.2.0.img.gz) or [ZCU104 board image](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-zcu104-dpu-v2020.1-v1.2.0.img.gz) and flash it to SD card (16GB recommanded) using Etcher or Win32DiskImager. Note that you may need to run command irps5401 first to trigger the power management patch for ZCU104 to avoid system hang or power off issue when running samples. Follow the steps below to run the samples. It should be noted that the above board images have placed DNNDK package and samples on path /home/root/Vitis-AI/mpsoc, you can skip steps 1 and 2 if you don't make changes.

1. Copy the package [vitis-ai_v1.2_dnndk.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai_v1.2_dnndk.tar.gz) to /home/root directory of the evaluation board, and follow the steps below to set up the environment on the board
```sh
tar -xzvf vitis-ai_v1.2_dnndk.tar.gz
cd vitis-ai_v1.2_dnndk
./install.sh
```

2. Copy the Vitis-AI/mpsoc/vitis_ai_dnndk_samples folder from host to the /home/root directory of the evaluation board

3. Download [vitis-ai_v1.2_dnndk_sample_img.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai_v1.2_dnndk_sample_img.tar.gz) package to /home/root of the evaluation board and run command below
```sh
tar -xzvf vitis-ai_v1.2_dnndk_sample_img.tar.gz
```

4. Run the samples on evaluation board
```sh
cd /home/root/vitis-ai_v1.2_dnndk_sample/resnet50
./resnet50

```
5. Run the other samples according to the following chapters of the Vitis AI User Guide, Quick Start -> Running Examples -> Legacy DNNDK Examples. For video input, only webm and raw format are supported by default with the above system image. If you want to support video data in other formats, you need to install the relevant packages on the system.

6. If you use your own system image, you may need to copy dpu.xclbin to /usr/lib first. Such as
```sh
cp /media/sd-mmcblk0p1/dpu.xclbin /usr/lib/
```

## References 
- [Vitis AI User Guide](https://www.xilinx.com/html_docs/vitis_ai/1_2/zkj1576857115470.html)
- [Vitis AI Model Zoo](https://github.com/Xilinx/Vitis-AI/tree/master/AI-Model-Zoo)

