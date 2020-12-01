<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Vitis AI Runtime v1.3</h1>
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
    │   └── src
    ├── common
    │   ├── common.cpp
    │   └── common.h
    ├── inception_v1_mt_py
    │   ├── inception_v1.py
    │   └── words.txt
    ├── pose_detection
    │   ├── build.sh
    │   └── src
    ├── resnet50
    │   ├── build.sh
    │   ├── src
    │   └── words.txt
    ├── resnet50_mt_py
    │   ├── resnet50.py
    │   └── words.txt
    ├── segmentation
    │   ├── build.sh
    │   └── src
    ├── squeezenet_pytorch
    │   ├── build.sh
    │   ├── src
    │   └── words.txt
    └── video_analysis
        ├── build.sh
        └── src

```

## Quick Start For Edge
### Setting Up the Host
1. Download the [sdk-2020.2.0.0.sh](https://www.xilinx.com/bin/public/openDownload?filename=sdk-2020.2.0.0.sh)

2. Install the cross-compilation system environment, follow the prompts to install. 

**Please install it on your local host linux system, not in the docker system.**
```
./sdk-2020.2.0.0.sh
```
Note that the `~/petalinux_sdk` path is recommended for the installation. Regardless of the path you choose for the installation, make sure the path has read-write permissions. 
Here we install it under `~/petalinux_sdk`.

3. When the installation is complete, follow the prompts and execute the following command.
```
source ~/petalinux_sdk/environment-setup-aarch64-xilinx-linux
```
Note that if you close the current terminal, you need to re-execute the above instructions in the new terminal interface.

4. Download the [vitis_ai_2020.2-r1.3.0.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_2020.2-r1.3.0.tar.gz) and install it to the petalinux system.
```
tar -xzvf vitis_ai_2020.2-r1.3.0.tar.gz -C ~/petalinux_sdk/sysroots/aarch64-xilinx-linux
```

5. Cross compile the sample, take `resnet50` as an example.
```
cd ~/Vitis-AI/examples/VART/samples/resnet50
bash -x build.sh
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
	
		[ZCU102](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-zcu102-dpu-v2020.2-v1.3.0.img.gz)  
	
		[ZCU104](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-zcu104-dpu-v2020.2-v1.3.0.img.gz)  
	
      	Note: The version of the board image should be 2020.2 or above.
	* Use Etcher software to burn the image file onto the SD card.
	* Insert the SD card with the image into the destination board.
	* Plug in the power and boot the board using the serial port to operate on the system.
	* Set up the IP information of the board using the serial port.
	You can now operate on the board using SSH.

2. (Optional) Running `zynqmp_dpu_optimize.sh` to optimize the board setting.
	
	The script runs automatically after the board boots up with the official image.
	But you can also download the `dpu_sw_optimize.tar.gz` from [here](../../DPU-TRD/app/dpu_sw_optimize.tar.gz).
	```
	cd ~/dpu_sw_optimize/zynqmp/
	./zynqmp_dpu_optimize.sh
	```	

3. (Optional) How to update Vitis AI Runtime and install them separately. 
	
	If you want to update the Vitis AI Runtime or install them to your custom board image, follow these steps.
	* Download the [Vitis AI Runtime 1.3.0](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-1.3.0.tar.gz).  	
	* Untar the runtime packet and copy the following folder to the board using scp.
	```
	tar -xzvf vitis-ai-runtime-1.3.0.tar.gz
	scp -r vitis-ai-runtime-1.3.0/aarch64/centos root@IP_OF_BOARD:~/
	```
	* Log in to the board using ssh. You can also use the serial port to login.
	* Install the Vitis AI Runtime. Execute the following command in order.
	```
	cd ~/centos
	rpm -ivh --force libunilog-1.3.0-r<x>.aarch64.rpm
	rpm -ivh --force libxir-1.3.0-r<x>.aarch64.rpm
	rpm -ivh --force libtarget-factory-1.3.0-r<x>.aarch64.rpm
	rpm -ivh --force libvart-1.3.0-r<x>.aarch64.rpm
	```
4. (Optional) Download the model.  	
	For each model, there will be a yaml file which is used for describe all the details about the model. 
	In the yaml, you will find the model's download links for different platforms. Please choose the corresponding model and download it.
	Click [Xilinx AI Model Zoo](../../models/AI-Model-Zoo/model-list) to view all the models.
	
	* Take `resnet50` of ZCU102 as an example.
	```
	  cd /workspace
	  wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-zcu102-zcu104-r1.3.0.tar.gz -O resnet50-zcu102-zcu104-r1.3.0.tar.gz
	```	
	* Copy the downloaded file to the board using scp with the following command. 
	```
	  scp resnet50-zcu102-zcu104-r1.3.0.tar.gz root@IP_OF_BOARD:~/
	```
	* Log in to the board (usong ssh or serial port) and install the model package.
	```
	  tar -xzvf resnet50-zcu102-zcu104-r1.3.0.tar.gz
	  cp resnet50 /usr/share/vitis_ai_library/models -r
	```
	  
### Running Vitis AI Examples

1. Download the [vitis_ai_runtime_r1.3.x_image_video.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_runtime_r1.3.0_image_video.tar.gz) from host to the target using scp with the following command.
	```
	[Host]$scp vitis_ai_runtime_r1.3.x_image_video.tar.gz root@[IP_OF_BOARD]:~/
	```
2. Unzip the `vitis_ai_runtime_r1.3.x_image_video.tar.gz` package on the target.
	```
	cd ~
	tar -xzvf vitis_ai_runtime_r*1.3*_image_video.tar.gz -C Vitis-AI/examples/VART
	```
3. Enter the directory of samples in the target board. Take `resnet50` as an example.
	```
	cd ~/Vitis-AI/examples/VART/samples/resnet50
	```
4. Run the example.
	```
	./resnet50 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel
	```

	For examples with video input, only `webm` and `raw` format are supported by default with the official system image. 
	If you want to support video data in other formats, you need to install the relevant packages on the system. 

 <summary><b>Launching Commands for VART Samples on edge </b></summary>
 
| No\. | Example Name             | Command                                                      |
| :--- | :----------------------- | :----------------------------------------------------------- |
| 1    | resnet50                 | ./resnet50 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel                              |
| 2    | resnet50_mt_py           | python3 resnet50.py 1 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel                    |
| 3    | inception_v1_mt_py       | python3 inception_v1.py 1 /usr/share/vitis_ai_library/models/inception_v1_tf/inception_v1_tf.xmodel               |
| 4    | pose_detection           | ./pose_detection video/pose.webm /usr/share/vitis_ai_library/models/sp_net/sp_net.xmodel /usr/share/vitis_ai_library/models/ssd_pedestrian_pruned_0_97/ssd_pedestrian_pruned_0_97.xmodel         |
| 5    | video_analysis           | ./video_analysis video/structure.webm /usr/share/vitis_ai_library/models/ssd_traffic_pruned_0_9/ssd_traffic_pruned_0_9.xmodel    |
| 6    | adas_detection           | ./adas_detection video/adas.webm /usr/share/vitis_ai_library/models/yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel         |
| 7    | segmentation             | ./segmentation video/traffic.webm /usr/share/vitis_ai_library/models/fpn/fpn.xmodel        |



## Quick Start For Alveo
### Setting Up the Host

1. Click [DPUCAHX8H -- the DPU for Alveo Accelerator Card with HBM](../../setup/alveo/u50_u50lv_u280/README.md#dpucahx8h----the-dpu-for-alveo-accelerator-card-with-hbm) to set up the Alveo Card.

2. Download the xclbin files from [here](https://www.xilinx.com/bin/public/openDownload?filename=alveo_xclbin-1.3.0.tar.gz). Untar it, choose the Alveo card and install it. Take `U50`
as an example.
```
cd /workspace
wget https://www.xilinx.com/bin/public/openDownload?filename=alveo_xclbin-1.3.0.tar.gz -O alveo_xclbin-1.3.0.tar.gz
tar -xzvf alveo_xclbin-1.3.0.tar.gz
cd alveo_xclbin-1.3.0/U50/6E300M
sudo cp dpu.xclbin hbm_address_assignment.txt /usr/lib
```
### Running Vitis AI Examples
Suppose you have downloaded `Vitis-AI`, entered `Vitis-AI` directory, and then started Docker. 
Thus, `VART` is located in the path of `/workspace/examples/VART/` in the docker system. 

**`/workspace/examples/VART/` is the path for the following example.**
 
If you encounter any path errors in running examples, check to see if you follow the steps above.

1. Download the [vitis_ai_runtime_r1.3.0_image_video.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_runtime_r1.3.0_image_video.tar.gz) package and unzip it.
	```
	cd /workspace/examples
	wget https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_runtime_r1.3.0_image_video.tar.gz -O vitis_ai_runtime_r1.3.0_image_video.tar.gz
	tar -xzvf vitis_ai_runtime_r*1.3*_image_video.tar.gz -C VART
	```
2. Download the model.  	
	For each model, there will be a yaml file which is used for describe all the details about the model. 
	In the yaml, you will find the model's download links for different platforms. Please choose the corresponding model and download it.
	Click [Xilinx AI Model Zoo](../../models/AI-Model-Zoo/model-list) to view all the models.
	
	* Take `resnet50` of U50 as an example.
	```
	  cd /workspace
	  wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-u50-r1.3.0.tar.gz -O resnet50-u50-r1.3.0.tar.gz
	```	
	* Install the model package.  
	If the `/usr/share/vitis_ai_library/models` folder does not exist, create it first.
	```
	  mkdir /usr/share/vitis_ai_library/models
	```
	Then install the model package.
	```
	  tar -xzvf resnet50-u50-r1.3.0.tar.gz
	  sudo cp resnet50 /usr/share/vitis_ai_library/models -r
	```

3. Compile the sample, take `resnet50` as an example.
	```
	cd /workspace/examples/VART/samples/resnet50
	bash -x build.sh
	```
4. Run the example, take `U50` platform as an example.
	```
	./resnet50 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel
	```
	**Note that different alveo cards correspond to different model files, which cannot be used alternately.** 


 <summary><b>Launching Commands for VART Samples on U50 </b></summary>
 
| No\. | Example Name             | Command                                                   |
| :--- | :----------------------- | :-------------------------------------------------------- |
| 1    | resnet50                 | ./resnet50 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel                            |
| 2    | resnet50_mt_py           | /usr/bin/python3 resnet50.py 1 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel          |
| 3    | inception_v1_mt_py       | /usr/bin/python3 inception_v1.py 1 /usr/share/vitis_ai_library/models/inception_v1_tf/inception_v1_tf.xmodel      |
| 4    | pose_detection           | ./pose_detection video/pose.webm /usr/share/vitis_ai_library/models/sp_net/sp_net.xmodel /usr/share/vitis_ai_library/models/ssd_pedestrian_pruned_0_97/ssd_pedestrian_pruned_0_97.xmodel         |
| 5    | video_analysis           | ./video_analysis video/structure.webm /usr/share/vitis_ai_library/models/ssd_traffic_pruned_0_9/ssd_traffic_pruned_0_9.xmodel    |
| 6    | adas_detection           | ./adas_detection video/adas.webm /usr/share/vitis_ai_library/models/yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel         |
| 7    | segmentation             | ./segmentation video/traffic.webm /usr/share/vitis_ai_library/models/fpn/fpn.xmodel        |


