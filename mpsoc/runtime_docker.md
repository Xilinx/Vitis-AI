# Vitis AI Runtime Docker

## 1. Prerequisite
- Docker service is available on host machine. For detailed steps, please refer to [Install Docker](/doc/install_docker/README.md) page
- Vitis runtime docker image is correctly loaded and launched. For detailed steps, please refer to [Load and Run Docker](/doc/install_docker/load_run_docker.md)
- All necessary files are placed in `/workspace` folder 

## 2. Examples

All the examples created by Vitis AI API are place under `/opt/vitis_ai/sample`. 

```
dnndk_samples/
├── adas_detection
├── face_detection
├── inception_v1
├── inception_v1_mt
├── inception_v1_mt_py
├── mini_resnet_py
├── mobilenet
├── mobilenet_mt
├── pose_detection
├── resnet50
├── resnet50_mt
├── segmentation
├── tf_resnet50
└── video_analysis

vitis_ai_samples/
├── adas_detection
├── inception_v1_mt_py
├── pose_detection
├── resnet50
├── resnet50_mt_py
├── segmentation
└── video_analysis
```

The runtime docker provides cross-compile environment to build application executables on host. Take Vitis AI API Resnet classification application (vitis_ai_samples_102/resnet50) as example. 

## 3. Build executable

Pre-compiled network elf file (dpu_resnet50.elf) is provided under `/resnet50/model`. If elf file needs to be changed, please copy corrsponding file(s) under same folder. 

Makefile, source code (/src/main.cc) and configuration file (dpuv2_rundir/meta.json/) are also provided as reference. 

To build executalbe simply with following command: 

`make`

```
/opt/vitis_ai/petalinux_sdk/sysroots/x86_64-petalinux-linux/usr/bin/aarch64-xilinx-linux/aarch64-xilinx-linux-g++ -c --sysroot=/opt/vitis_ai/petalinux_sdk/sysroots/aarch64-xilinx-linux -O2 -Wall -Wpointer-arith -std=c++11 -ffast-math -I/opt/vitis_ai/petalinux_sdk/sysroots/aarch64-xilinx-linux/usr/include -I/opt/vitis_ai/petalinux_sdk/sysroots/aarch64-xilinx-linux/usr/local/include -I./include -I../common/ -mcpu=cortex-a53 /opt/vitis_ai/sample/vitis_ai_samples_zcu102/resnet50/src/main.cc -o /opt/vitis_ai/sample/vitis_ai_samples_zcu102/resnet50/build/main.o
/opt/vitis_ai/petalinux_sdk/sysroots/x86_64-petalinux-linux/usr/bin/aarch64-xilinx-linux/aarch64-xilinx-linux-g++ -c --sysroot=/opt/vitis_ai/petalinux_sdk/sysroots/aarch64-xilinx-linux -O2 -Wall -Wpointer-arith -std=c++11 -ffast-math -I/opt/vitis_ai/petalinux_sdk/sysroots/aarch64-xilinx-linux/usr/include -I/opt/vitis_ai/petalinux_sdk/sysroots/aarch64-xilinx-linux/usr/local/include -I./include -I../common/ -mcpu=cortex-a53 /opt/vitis_ai/sample/vitis_ai_samples_zcu102/common/common.cpp -o /opt/vitis_ai/sample/vitis_ai_samples_zcu102/resnet50/build/common.o
/opt/vitis_ai/petalinux_sdk/sysroots/x86_64-petalinux-linux/usr/bin/aarch64-xilinx-linux/aarch64-xilinx-linux-g++ --sysroot=/opt/vitis_ai/petalinux_sdk/sysroots/aarch64-xilinx-linux -O2 -Wall -Wpointer-arith -std=c++11 -ffast-math -I/opt/vitis_ai/petalinux_sdk/sysroots/aarch64-xilinx-linux/usr/include -I/opt/vitis_ai/petalinux_sdk/sysroots/aarch64-xilinx-linux/usr/local/include -I./include -I../common/ -mcpu=cortex-a53 /opt/vitis_ai/sample/vitis_ai_samples_zcu102/resnet50/build/main.o /opt/vitis_ai/sample/vitis_ai_samples_zcu102/resnet50/build/common.o /opt/vitis_ai/sample/vitis_ai_samples_zcu102/resnet50/model/dpu_resnet50.elf -o resnet50 --sysroot=/opt/vitis_ai/petalinux_sdk/sysroots/aarch64-xilinx-linux -L/opt/vitis_ai/petalinux_sdk/sysroots/aarch64-xilinx-linux/usr/local/lib/ -L/opt/vitis_ai/petalinux_sdk/sysroots/aarch64-xilinx-linux/lib -L/opt/vitis_ai/petalinux_sdk/sysroots/aarch64-xilinx-linux/usr/lib  -ln2cube -lhineon -lopencv_videoio  -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_core -lpthread

```

After it finished, the executalbe `renset50` is generated. 

Boot ZCU102 or ZCU104 board using flashed SD card with the prebuilt board image.   

[ZCU102 board image](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-zcu102-dpu-v2019.2.img.gz)  

[ZCU104 board image](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-zcu104-dpu-v2019.2.img.gz)

Copy the execuatle and necessary file on to board and run. 

For DNNDK example, the files might be different but the procedures are identical.   

For more information, please refer to [UG1414](http://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_0/ug1414-vitis-ai.pdf).

