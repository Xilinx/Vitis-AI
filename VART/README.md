<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Vitis AI Runtime v1.2</h1>
    </td>
 </tr>
 </table>

# Introduction
Vitis AI Run time enables applications to use the unified high-level runtime API for both cloud and edge. Therefore, making cloud-to-edge deployments seamless and efficient.
The Vitis AI Runtime API features are:
* Asynchronous submission of jobs to the accelerator
* Asynchronous collection of jobs from the accelerator
* C++ and Python implementations
* Support for multi-threading and multi-process execution

For edge users, click 
[Quick Start For Edge](#quick-start-for-edge) to get started quickly. 

For cloud users, click 
[Quick Start For Alveo](#quick-start-for-alveo) to get started quickly.

Vitis AI Runtime directory structure introduction
--------------------------------------------------

```
VART
├── README.md
└── samples
    ├── adas_detection
    │   ├── build.sh
    │   ├── model_dir_for_U280
    │   ├── model_dir_for_U50
    │   ├── model_dir_for_U50lv_10E
    │   ├── model_dir_for_U50lv_9E
    │   ├── model_dir_for_zcu102
    │   ├── model_dir_for_zcu104
    │   └── src
    ├── common
    │   ├── common.cpp
    │   └── common.h
    ├── inception_v1_mt_py
    │   ├── inception_v1.py
    │   ├── input_fn.py
    │   ├── model_dir_for_U280
    │   ├── model_dir_for_U50
    │   ├── model_dir_for_U50lv_10E
    │   ├── model_dir_for_U50lv_9E
    │   ├── model_dir_for_zcu102
    │   └── model_dir_for_zcu104
    ├── pose_detection
    │   ├── build.sh
    │   ├── model_dir_for_U280
    │   ├── model_dir_for_U50
    │   ├── model_dir_for_U50lv_10E
    │   ├── model_dir_for_U50lv_9E
    │   ├── model_dir_for_zcu102
    │   ├── model_dir_for_zcu104
    │   └── src
    ├── resnet50
    │   ├── build.sh
    │   ├── model_dir_for_U280
    │   ├── model_dir_for_U50
    │   ├── model_dir_for_U50lv_10E
    │   ├── model_dir_for_U50lv_9E
    │   ├── model_dir_for_zcu102
    │   ├── model_dir_for_zcu104
    │   ├── src
    │   └── words.txt
    ├── resnet50_mt_py
    │   ├── input_fn.py
    │   ├── input.py
    │   ├── model_dir_for_U280
    │   ├── model_dir_for_U50
    │   ├── model_dir_for_U50lv_10E
    │   ├── model_dir_for_U50lv_9E
    │   ├── model_dir_for_zcu102
    │   ├── model_dir_for_zcu104
    │   └── resnet50.py
    ├── segmentation
    │   ├── build.sh
    │   ├── model_dir_for_U280
    │   ├── model_dir_for_U50
    │   ├── model_dir_for_U50lv_10E
    │   ├── model_dir_for_U50lv_9E
    │   ├── model_dir_for_zcu102
    │   ├── model_dir_for_zcu104
    │   └── src
    └── video_analysis
        ├── build.sh
        ├── model_dir_for_U280
        ├── model_dir_for_U50
        ├── model_dir_for_U50lv_10E
        ├── model_dir_for_U50lv_9E
        ├── model_dir_for_zcu102
        ├── model_dir_for_zcu104
        └── src

```

## Quick Start For Edge
### Setting Up the Host
1. Download the [sdk-2020.1.0.0.sh](https://www.xilinx.com/bin/public/openDownload?filename=sdk-2020.1.0.0.sh)

2. Install the cross-compilation system environment, follow the prompts to install. 

**Please install it on your local host linux system, not in the docker system.**
```
$./sdk-2020.1.0.0.sh
```
Note that the `~/petalinux_sdk` path is recommended for the installation. Regardless of the path you choose for the installation, make sure the path has read-write permissions. 
Here we install it under `~/petalinux_sdk`.

3. When the installation is complete, follow the prompts and execute the following command.
```
$source ~/petalinux_sdk/environment-setup-aarch64-xilinx-linux
```
Note that if you close the current terminal, you need to re-execute the above instructions in the new terminal interface.

4. Download the [vitis_ai_2020.1-r1.2.0.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_2020.1-r1.2.0.tar.gz) and install it to the petalinux system.
```
$tar -xzvf vitis_ai_2020.1-r1.2.0.tar.gz -C ~/petalinux_sdk/sysroots/aarch64-xilinx-linux
```

5. Cross compile the sample, take resnet50 as an example.
```
$cd ~/Vitis-AI/VART/samples/resnet50
$bash -x build.sh
```	
If the compilation process does not report any error and the executable file `resnet50` is generated, the host environment is installed correctly.

### Setting Up the Target

**To improve the user experience, the Vitis AI Runtime packages, VART samples, Vitis-AI-Library samples and
models have been built into the board image. Therefore, user does not need to install Vitis AI
Runtime packages and model package on the board separately. However, users can still install
the model or Vitis AI Runtime on their own image or on the official image by following these
steps.**

1. Installing a Board Image.
	* Download the SD card system image files from the following links:  
	
		[ZCU102](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-zcu102-dpu-v2020.1-v1.2.0.img.gz)  
	
		[ZCU104](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-zcu104-dpu-v2020.1-v1.2.0.img.gz)  
	
      	Note: The version of the board image should be 2020.1 or above.
	* Use Etcher software to burn the image file onto the SD card.
	* Insert the SD card with the image into the destination board.
	* Plug in the power and boot the board using the serial port to operate on the system.
	* Set up the IP information of the board using the serial port.
	You can now operate on the board using SSH.

2. (Optional) Running `zynqmp_dpu_optimize.sh` to optimize the board setting.
	
	The script runs automatically after the board boots up with the official image.
	But you can also download the `dpu_sw_optimize.tar.gz` from [here](../DPU-TRD/app/dpu_sw_optimize.tar.gz).
	```
	#cd ~/dpu_sw_optimize/zynqmp/
	#./zynqmp_dpu_optimize.sh
	```	

3. (Optional) How to update Vitis AI Runtime and install them separately. 
	
	If you want to update the Vitis AI Runtime or install them to your custom board image, follow these steps.
	* Download the [Vitis AI Runtime 1.2.1](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-1.2.1.tar.gz).  	
	* Untar the runtime packet and copy the following folder to the board using scp.
	```
	$tar -xzvf vitis-ai-runtime-1.2.1.tar.gz
	$scp -r vitis-ai-runtime-1.2.1/aarch64/centos root@IP_OF_BOARD:~/
	```
	* Log in to the board using ssh. You can also use the serial port to login.
	* Install the Vitis AI Runtime. Execute the following command in order.
	```
	#cd ~/centos
	#rpm -ivh --force libunilog-1.2.0-r<x>.aarch64.rpm
	#rpm -ivh --force libxir-1.2.0-r<x>.aarch64.rpm
	#rpm -ivh --force libtarget-factory-1.2.0-r<x>.aarch64.rpm
	#rpm -ivh --force libvart-1.2.0-r<x>.aarch64.rpm
	```
	  
### Running Vitis AI Examples

1. Download the [vitis_ai_runtime_r1.2.x_image_video.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_runtime_r1.2.0_image_video.tar.gz) from host to the target using scp with the following command.
	```
	[Host]$scp vitis_ai_runtime_r1.2.x_image_video.tar.gz root@[IP_OF_BOARD]:~/
	```
2. Unzip the `vitis_ai_runtime_r1.2.x_image_video.tar.gz` package on the target.
	```
	#cd ~
	#tar -xzvf vitis_ai_runtime_r*1.2*_image_video.tar.gz -C Vitis-AI/VART
	```
3. Enter the directory of samples in the target board. Take `resnet50` as an example.
	```
	#cd ~/Vitis-AI/VART/samples/resnet50
	```
5. Run the example.

	For ZCU102, execute the following command.
	```
	#./resnet50 model_dir_for_zcu102/resnet50.elf
	```
	For ZCU104, execute the following command.
	```
	#./resnet50 model_dir_for_zcu104/resnet50.elf
	```

	For examples with video input, only `webm` and `raw` format are supported by default with the official system image. 
	If you want to support video data in other formats, you need to install the relevant packages on the system. 

 <summary><b>Launching Commands for VART Samples on ZCU102 </b></summary>
 
| No\. | Example Name             | Command                                                      |
| :--- | :----------------------- | :----------------------------------------------------------- |
| 1    | resnet50                 | ./resnet50 model_dir_for_zcu102/resnet50.elf                              |
| 2    | resnet50_mt_py           | python3 resnet50.py 1 model_dir_for_zcu102/resnet50.elf                    |
| 3    | inception_v1_mt_py       | python3 inception_v1.py 1 model_dir_for_zcu102/inception_v1_tf.elf               |
| 4    | pose_detection           | ./pose_detection video/pose.webm model_dir_for_zcu102/sp_net.elf model_dir_for_zcu102/ssd_pedestrain_pruned_0_97.elf         |
| 5    | video_analysis           | ./video_analysis video/structure.webm model_dir_for_zcu102/ssd_traffic_pruned_0_9.elf    |
| 6    | adas_detection           | ./adas_detection video/adas.webm model_dir_for_zcu102/yolov3_adas_pruned_0_9.elf         |
| 7    | segmentation             | ./segmentation video/traffic.webm model_dir_for_zcu102/fpn.elf        |



## Quick Start For Alveo
### Setting Up the Host

Click [DPUCAHX8H -- the DPU for Alveo Accelerator Card with HBM](../alveo-hbm#dpucahx8h----the-dpu-for-alveo-accelerator-card-with-hbm) to set up the Alveo Card.

### Running Vitis AI Examples
Suppose you have downloaded `Vitis-AI`, entered `Vitis-AI` directory, and then started Docker. 
Thus, `VART` is located in the path of `/workspace/VART/` in the docker system. 

**`/workspace/VART/` is the path for the following example.**
 
If you encounter any path errors in running examples, check to see if you follow the steps above.

1. Download the [vitis_ai_runtime_r1.2.0_image_video.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_runtime_r1.2.0_image_video.tar.gz) package and unzip it.
	```
	$cd /workspace
	$wget https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_runtime_r1.2.0_image_video.tar.gz -O vitis_ai_runtime_r1.2.0_image_video.tar.gz
	$tar -xzvf vitis_ai_runtime_r*1.2*_image_video.tar.gz -C VART
	```
2. Compile the sample, take `resnet50` as an example.
	```
	$cd /workspace/VART/samples/resnet50
	$bash -x build.sh
	```
3. Run the example, take `U50` platform as an example.
	```
	$./resnet50 model_dir_for_U50/resnet50.xmodel
	```
	**Note that different alveo cards correspond to different model files, which cannot be used alternately.** 


 <summary><b>Launching Commands for VART Samples on U50 </b></summary>
 
| No\. | Example Name             | Command                                                   |
| :--- | :----------------------- | :-------------------------------------------------------- |
| 1    | resnet50                 | ./resnet50 model_dir_for_U50/resnet50.xmodel                             |
| 2    | resnet50_mt_py           | /usr/bin/python3 resnet50.py 1 model_dir_for_U50/resnet50.xmodel          |
| 3    | inception_v1_mt_py       | /usr/bin/python3 inception_v1.py 1 model_dir_for_U50/inception_v1_tf.xmodel      |
| 4    | pose_detection           | ./pose_detection video/pose.mp4 model_dir_for_U50/sp_net.xmodel model_dir_for_U50/ssd_pedestrain_pruned_0_97.xmodel          |
| 5    | video_analysis           | ./video_analysis video/structure.mp4 model_dir_for_U50/ssd_traffic_pruned_0_9.xmodel    |
| 6    | adas_detection           | ./adas_detection video/adas.avi model_dir_for_U50/yolov3_adas_pruned_0_9.xmodel         |
| 7    | segmentation             | ./segmentation video/traffic.mp4 model_dir_for_U50/fpn.xmodel        |


