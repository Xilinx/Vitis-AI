<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Vitis AI v2.5</h1>
    </td>
 </tr>
 </table>

# Introduction
This directory contains instructions for running DPUCZDX8G on Zynq Ultrascale+ MPSoC platforms.
**DPUCZDX8G**  is a configurable computation engine dedicated for convolutional neural networks.
It includes a set of highly optimized instructions, and supports most convolutional neural networks, such as VGG, ResNet, GoogleNet, YOLO, SSD, MobileNet, FPN, and others.
With Vitis-AI, Xilinx has integrated all the edge and cloud solutions under a unified API and toolset.

## Step1: Setup cross-compiler
1. Run the following command to install cross-compilation system environment.

**Please install it on your local host linux system, not in the docker system.**
```
./host_cross_compiler_setup.sh
```
Note that the Cross Compiler will be installed in `~/petalinux_sdk_2022.1` by default. 

2. When the installation is complete, follow the prompts and execute the following command.
```
source ~/petalinux_sdk_2022.1/environment-setup-cortexa72-cortexa53-xilinx-linux
```
Note that if you close the current terminal, you need to re-execute the above instructions in the new terminal interface.


## Step2: Setup the Target

**To improve the user experience, the Vitis AI Runtime packages, VART samples, Vitis-AI-Library samples and
models have been built into the board image. Therefore, user does not need to install Vitis AI
Runtime packages and model package on the board separately. However, users can still install
the model or Vitis AI Runtime on their own image or on the official image by following these
steps.**

1. Installing a Board Image.
	* Download the SD card system image files from the following links:  
	
		[ZCU102](https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-zcu102-dpu-v2022.1-v2.5.0.img.gz)  
	
		[ZCU104](https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-zcu104-dpu-v2022.1-v2.5.0.img.gz)  
		
		[KV260](https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-kv260-dpu-v2022.1-v2.5.0.img.gz) 
		
	
      	Note: For ZCU102/ZCU104/KV260, the version of the board image should be 2022.1 or above.  

	* Use Etcher software to burn the image file onto the SD card.
	* Insert the SD card with the image into the destination board.
	* Plug in the power and boot the board using the serial port to operate on the system.
	* Set up the IP information of the board using the serial port.
	
	**For the details, please refer to [Setting Up the Evaluation Board](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Setting-Up-the-Evaluation-Board)**

2. (Optional) Running `zynqmp_dpu_optimize.sh` to optimize the board setting.
	
	The script runs automatically after the board boots up with the official image.
	But you can also find the `dpu_sw_optimize.tar.gz` in [DPUCZDX8G.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=DPUCZDX8G.tar.gz).
	```
	cd ~/dpu_sw_optimize/zynqmp/
	./zynqmp_dpu_optimize.sh
	```	

3. (Optional) How to install the Vitis AI for PetaLinux 2022.1  
	There are two ways to install the dependent libraries of Vitis-AI. One is to rebuild the system by configuring PetaLinux and the other is to install the Vitis-AI online via `dnf`.
	* Build-Time  
	  For `VAI2.5 Recipes`, refer to [Vitis-AI-Recipes](../petalinux).  
	* Run-Time   
	  Execute `dnf install packagegroup-petalinux-vitisai` to complete the installation on the target. For more details, refer to [VAI2.5 Online Install](../petalinux#to-install-the-vai25-online) 
	
4. (Optional) How to update Vitis AI Runtime and install them separately.  
    If you want to update the Vitis AI Runtime or install them to your custom board image, follow these steps.
	* Copy the following folder to the board using scp.
	```
	scp -r mpsoc root@IP_OF_BOARD:~/
	```
	* Log in to the board using ssh. You can also use the serial port to login.
	* Install the Vitis AI Runtime. Execute the following command.
	```
	cd ~/mpsoc
	bash target_vart_setup.sh
	```
5. (Optional) Download the model.  	
	For each model, there will be a yaml file which is used for describe all the details about the model. 
	In the yaml, you will find the model's download links for different platforms. Please choose the corresponding model and download it.
	Click [Xilinx AI Model Zoo](../../model_zoo/model-list) to view all the models.
	
	* Take `resnet50` of ZCU102 as an example.
	```
	  cd /workspace
	  wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-zcu102_zcu104_kv260-r2.5.0.tar.gz -O resnet50-zcu102_zcu104_kv260-r2.5.0.tar.gz
	```	
	* Copy the downloaded file to the board using scp with the following command. 
	```
	  scp resnet50-zcu102_zcu104_kv260-r2.5.0.tar.gz root@IP_OF_BOARD:~/
	```
	* Log in to the board (using ssh or serial port) and install the model package.
	```
	  tar -xzvf resnet50-zcu102_zcu104_kv260-r2.5.0.tar.gz
	  cp resnet50 /usr/share/vitis_ai_library/models -r
	```
	  
## Step3: Run the Vitis AI Examples

1. Download the [vitis_ai_runtime_r2.5.x_image_video.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_runtime_r2.5.0_image_video.tar.gz) from host to the target using scp with the following command.
	```
	[Host]$scp vitis_ai_runtime_r2.5.*_image_video.tar.gz root@[IP_OF_BOARD]:~/
	```
2. Unzip the `vitis_ai_runtime_r2.5.x_image_video.tar.gz` package on the target.
	```
	cd ~
	tar -xzvf vitis_ai_runtime_r*2.5*_image_video.tar.gz -C Vitis-AI/examples/VART
	```
3. Enter the directory of samples in the target board. Take `resnet50` as an example.
	```
	cd ~/Vitis-AI/examples/VART/resnet50
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
| 2    | resnet50_pt              | ./resnet50_pt /usr/share/vitis_ai_library/models/resnet50_pt/resnet50_pt.xmodel ../images/001.jpg                           |
| 3    | resnet50_ext             | ./resnet50_ext /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel ../images/001.jpg                           |
| 4    | resnet50_mt_py           | python3 resnet50.py 1 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel                    |
| 5    | inception_v1_mt_py       | python3 inception_v1.py 1 /usr/share/vitis_ai_library/models/inception_v1_tf/inception_v1_tf.xmodel               |
| 6    | pose_detection           | ./pose_detection video/pose.webm /usr/share/vitis_ai_library/models/sp_net/sp_net.xmodel /usr/share/vitis_ai_library/models/ssd_pedestrian_pruned_0_97/ssd_pedestrian_pruned_0_97.xmodel         |
| 7    | video_analysis           | ./video_analysis video/structure.webm /usr/share/vitis_ai_library/models/ssd_traffic_pruned_0_9/ssd_traffic_pruned_0_9.xmodel    |
| 8    | adas_detection           | ./adas_detection video/adas.webm /usr/share/vitis_ai_library/models/yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel         |
| 9    | segmentation             | ./segmentation video/traffic.webm /usr/share/vitis_ai_library/models/fpn/fpn.xmodel        |
| 10   | squeezenet_pytorch       | ./squeezenet_pytorch /usr/share/vitis_ai_library/models/squeezenet_pt/squeezenet_pt.xmodel        |

## References
- [Vitis AI User Guide](https://www.xilinx.com/html_docs/vitis_ai/2_5/index.html)
