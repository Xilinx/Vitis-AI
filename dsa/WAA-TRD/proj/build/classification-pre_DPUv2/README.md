# Classification example: TRD run using Pre-processor & DPU source files

## 1 Generate SD card image

```
% cd $TRD_HOME/proj/build/classification-pre_DPUv2
% ./build_classification_pre.sh
```
Note that 
- Generated SD card image will be here **$TRD_HOME/proj/build/classification-pre_DPUv2/binary_container_1/sd_card.img**.
- The default setting of DPU is **B4096** with RAM_USAGE_LOW, CHANNEL_AUGMENTATION_ENABLE, DWCV_ENABLE, POOL_AVG_ENABLE, RELU_LEAKYRELU_RELU6, Softmax. Modify the `$TRD_HOME/proj/build/classification-pre_DPUv2/dpu_conf.vh` file can change the default settings.
- Build runtime is ~4.5 hours

## 2 Installing board image
- Use Etcher software to burn the sd card image file onto the SD card.


## 3 Installing Vitis AI Runtime on the Evaluation Board

- Download the [Vitis AI Runtime 1.3.0](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-1.3.0.tar.gz)  
	
- Untar the runtime packet and copy the following folder to the board using scp.
```
	tar -xzvf vitis-ai-runtime-1.3.0.tar.gz
	scp -r vitis-ai-runtime-1.3.0/aarch64/centos root@IP_OF_BOARD:~/
```
- Log in to the board using ssh. You can also use the serial port to login.
- Install the Vitis AI Runtime. Execute the following command in order.
```
	cd ~/centos
    rpm2cpio libvart-1.3.0-r<x>.aarch64 | cpio -idmv
	rpm -ivh --force libunilog-1.3.0-r<x>.aarch64.rpm
	rpm -ivh --force libxir-1.3.0-r<x>.aarch64.rpm
	rpm -ivh --force libtarget-factory-1.3.0-r<x>.aarch64.rpm
	rpm -ivh --force libvart-1.3.0-r<x>.aarch64.rpm
	rpm -ivh --force libvitis_ai_library-1.3.0-r<x>.aarch64.rpm
```
## 4 Run Resnet50 Example
This part is about how to run the Resnet50 example on zcu102 board.

Copy any image from `Vitis-AI/dsa/DPU-TRD/app/img/` and copy to `Vitis-AI/dsa/WAA-TRD/app/resnet50_waa/img` 

Copy the directory $TRD_HOME/app/resnet50_waa to the BOOT partition of the SD Card.

Pealse insert SD_CARD on the ZCU102 board. After the linux boot, run:

```
% cd /mnt/sd-mmcblk0p1/resnet50_waa
% cp /mnt/sd-mmcblk0p1/dpu.xclbin /usr/lib/
% export XILINX_XRT=/usr
% ./resnet50_waa model/resnet50.xmodel

Expect: 
Image : ./img/bellpeppe-994958.JPEG
top[0] prob = 0.990457  name = bell pepper
top[1] prob = 0.004048  name = acorn squash
top[2] prob = 0.002455  name = cucumber, cuke
top[3] prob = 0.000903  name = zucchini, courgette
top[4] prob = 0.000703  name = strawberry

```