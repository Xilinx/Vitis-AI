<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Vitis AI v2.5</h1>
    </td>
 </tr>
 </table>

# Introduction
This directory contains instructions for running DPUCVDX8G on Versal AI Core platforms.
**DPUCVDX8G**  is a configurable computation engine dedicated for convolutional neural networks.
It includes a set of highly optimized instructions, and supports most convolutional neural networks, such as VGG, ResNet, GoogleNet, YOLO, SSD, MobileNet, FPN, and others.
With Vitis-AI, Xilinx has integrated all the edge and cloud solutions under a unified API and toolset.

## Step1: Setup cross-compiler
1. Run the following command to install cross-compilation system environment.

**Please install it on your local host linux system, not in the docker system.**
```
./host_cross_compiler_setup.sh
```
Note that the Cross Compiler will be installed in `~/petalinux_sdk_2022.1` by default.  
For `VCK190 Production` board, use `host_cross_compiler_setup.sh` to install the cross-compiler. 

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
		
		[VCK190 Production board](https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-vck190-dpu-v2022.1-v2.5.0.img.gz)   
	
      	Note: The version of the VCK190 production board image is 2022.1.  

	* Use Etcher software to burn the image file onto the SD card.
	* Insert the SD card with the image into the destination board.
	* Plug in the power and boot the board using the serial port to operate on the system.
	* Set up the IP information of the board using the serial port.
	
	**For the details, please refer to [Setting Up the Evaluation Board](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Setting-Up-the-Evaluation-Board)**

2. (Optional) How to install the Vitis AI for PetaLinux 2022.1  
	There are two ways to install the dependent libraries of Vitis-AI. One is to rebuild the system by configuring PetaLinux and the other is to install the Vitis-AI online via `dnf`.
	* Build-Time
	  For `VAI2.5 Recipes`, refer to [Vitis-AI-Recipes](../petalinux).
	* Run-Time
	  Execute `dnf install packagegroup-petalinux-vitisai` to complete the installation on the target. For more details, refer to [VAI2.5 Online Install](../petalinux#to-install-the-vai25-online)
	
3. (Optional) How to update Vitis AI Runtime and install them separately. 
	
	If you want to update the Vitis AI Runtime or install them to your custom board image, follow these steps.
	* Copy the following folder to the board using scp.
	```
	scp -r vck190 root@IP_OF_BOARD:~/
	```
	* Log in to the board using ssh. You can also use the serial port to login.
	* Install the Vitis AI Runtime. Execute the following command.
	```
	cd ~/vck190
	bash target_vart_setup.sh
	```
4. (Optional) Download the model.  	
	For each model, there will be a yaml file which is used for describe all the details about the model. 
	In the yaml, you will find the model's download links for different platforms. Please choose the corresponding model and download it.
	Click [Xilinx AI Model Zoo](../../model_zoo/model-list) to view all the models.
	
	* Take `resnet50` of VCK190 as an example.
	```
	  cd /workspace
	  wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-vck190-r2.5.0.tar.gz -O resnet50-vck190-r2.5.0.tar.gz
	```	
	* Copy the downloaded file to the board using scp with the following command. 
	```
	  scp resnet50-vck190-r2.5.0.tar.gz root@IP_OF_BOARD:~/
	```
	* Log in to the board (using ssh or serial port) and install the model package.
	```
	  tar -xzvf resnet50-vck190-r2.5.0.tar.gz
	  cp resnet50 /usr/share/vitis_ai_library/models -r
	```
	  
## Step3: Run the Vitis AI Examples
Follow [Running Vitis AI Examples](../mpsoc/README.md#step3-run-the-vitis-ai-examples) to run Vitis AI examples.

## References
- [Vitis AI User Guide](https://www.xilinx.com/html_docs/vitis_ai/2_5/index.html)
