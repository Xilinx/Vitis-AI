# Detection example: TRD run Pre-processor & DPU source files

## 1 Generate SD card image

```
% cd $TRD_HOME/proj/build/detection-pre_DPUv2
% ./build_detection_pre.sh 
```
Note that
- Generated SD card image will be here **$TRD_HOME/proj/build/detection-pre_DPUv2/binary_container_1/sd_card.img**.
- The default setting of DPU is **B4096** with RAM_USAGE_LOW, CHANNEL_AUGMENTATION_ENABLE, DWCV_ENABLE, POOL_AVG_ENABLE, RELU_LEAKYRELU_RELU6, Softmax. Modify the `$TRD_HOME/proj/build/detection-pre_DPUv2/dpu_conf.vh` file can change the default settings.
- Build runtime is ~4.5 hours

### 2 Installing board image
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
## 4 Run Adas detection Example
This part is about how to run the Adas detection example on zcu102 board.

Download the images at https://cocodataset.org/#download. Please select suitable images which has car, bicycle or pedestrian and copy these images to `Vitis-AI/dsa/WAA-TRD/app/adas_detection_waa/data`. 

Copy the directory $TRD_HOME/app/adas_detection_waa to the BOOT partition of the SD Card.

Pealse insert SD_CARD on the ZCU102 board. After the linux boot, run:

```
% cd /mnt/sd-mmcblk0p1/adas_detection_waa
% export XILINX_XRT=/usr
% cp /mnt/sd-mmcblk0p1/dpu.xclbin /usr/lib/
% mkdir output
% ./adas_detection_waa model/yolov3_adas_pruned_0_9.xmodel

Expect: 
Input Image:./data/img1.jpg
Output Image:./output/img1.jpg

```