<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Vitis AI Library v1.2</h1>
    </td>
 </tr>
 </table>

# Introduction
The Vitis AI Library is a set of high-level libraries and APIs built for efficient AI inference with Deep-Learning Processor Unit (DPU). It is built based on the Vitis AI Runtime with Unified APIs, and it fully supports XRT 2019.2.

The Vitis AI Library provides an easy-to-use and unified interface by encapsulating many efficient and high-quality neural networks. This simplifies the use of deep-learning neural networks, even for users without knowledge of deep-learning or FPGAs. The Vitis AI Library allows users to focus more on the development of their applications, rather than the underlying hardware.


<p align="center">
  <img src="ai_library_diagram.png" >
</p>

For edge users, click 
[Quick Start For Edge](#quick-start-for-edge) to get started quickly. 

For cloud users, click 
[Quick Start For Alveo](#quick-start-for-alveo) to get started quickly.

Vitis AI Library directory structure introduction
--------------------------------------------------

```
vitis_ai_library
├── benchmark
│   ├── CMakeLists.txt
│   ├── include
│   └── src
├── classification
│   ├── CMakeLists.txt
│   ├── include
│   ├── samples
│   ├── src
│   └── test
├── cmake
│   ├── config.cmake.in
│   ├── XilinxCommon.cmake
│   ├── XilinxDpu.cmake
│   ├── xilinx_version.c.in
│   └── XilinxVersion.cmake
├── CMakeLists.txt
├── cmake.sh
├── Copyright.txt
├── dpu_task
│   ├── CMakeLists.txt
│   ├── Doxyfile
│   ├── include
│   ├── src
│   ├── test
│   └── util
├── facedetect
│   ├── CMakeLists.txt
│   ├── include
│   ├── src
│   └── test
├── facefeature
│   ├── CMakeLists.txt
│   ├── include
│   ├── src
│   └── test
├── facelandmark
│   ├── CMakeLists.txt
│   ├── include
│   ├── src
│   └── test
├── general
│   ├── CMakeLists.txt
│   ├── include
│   ├── src
│   └── test
├── lanedetect
│   ├── CMakeLists.txt
│   ├── include
│   ├── src
│   └── test
├── math
│   ├── buildme
│   ├── buildtest
│   ├── CMakeLists.txt
│   ├── cscope.files
│   ├── include
│   ├── src
│   ├── submodule
│   ├── test
│   └── update_submodule.sh
├── medicalsegmentation
│   ├── CMakeLists.txt
│   ├── include
│   ├── src
│   └── test
├── model_config
│   ├── CMakeLists.txt
│   ├── include
│   ├── src
│   └── test
├── multitask
│   ├── CMakeLists.txt
│   ├── include
│   ├── src
│   └── test
├── openpose
│   ├── CMakeLists.txt
│   ├── include
│   ├── src
│   └── test
├── overview
│   ├── cmake
│   ├── CMakeLists.txt
│   ├── demo							#AI Library demo
│   │   ├── classification
│   │   ├── seg_and_pose_detect
│   │   ├── segs_and_roadline_detect
│   │   └── yolov3
│   └── samples							#AI Library samples
│       ├── classification
│       ├── facedetect
│       ├── facefeature
│       ├── facelandmark
│       ├── lanedetect
│       ├── medicalsegmentation
│       ├── multitask
│       ├── openpose
│       ├── platedetect
│       ├── platenum
│       ├── posedetect
│       ├── refinedet
│       ├── reid
│       ├── segmentation
│       ├── ssd
│       ├── tfssd
│       ├── yolov2
│       └── yolov3
├── platedetect
│   ├── CMakeLists.txt
│   ├── include
│   ├── src
│   └── test
├── platenum
│   ├── CMakeLists.txt
│   ├── include
│   ├── src
│   └── test
├── platerecog
│   ├── CMakeLists.txt
│   ├── include
│   ├── src
│   └── test
├── posedetect
│   ├── CMakeLists.txt
│   ├── include
│   ├── src
│   └── test
├── refinedet
│   ├── CMakeLists.txt
│   ├── include
│   ├── src
│   └── test
├── reid
│   ├── CMakeLists.txt
│   ├── include
│   ├── src
│   └── test
├── segmentation
│   ├── CMakeLists.txt
│   ├── include
│   ├── src
│   └── test
├── ssd
│   ├── CMakeLists.txt
│   ├── include
│   ├── src
│   └── test
├── tfssd
│   ├── CMakeLists.txt
│   ├── include
│   ├── src
│   └── test
├── tracker
│   ├── CMakeLists.txt
│   ├── include
│   ├── src
│   └── test
├── usefultools
│   ├── CMakeLists.txt
│   ├── readme
│   ├── settings.sh.in
│   └── src
├── xnnpp
│   ├── CMakeLists.txt
│   ├── cscope.files
│   ├── include
│   └── src
├── yolov2
│   ├── CMakeLists.txt
│   ├── include
│   ├── src
│   └── test
└── yolov3
    ├── CMakeLists.txt
    ├── include
    ├── src
    └── test
```

## Quick Start For Edge
### Setting Up the Host
1. Download the [sdk-2020.1.0.0.sh](https://www.xilinx.com/bin/public/openDownload?filename=sdk-2020.1.0.0.sh)

2. Install the cross-compilation system environment, follow the prompts to install. 
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

5. Cross compile the demo in the AI Library, take `yolov3` as example.
```
$cd ~/Vitis-AI/Vitis-AI-Library/overview/demo/yolov3
$bash -x build.sh
```	

6. To compile the library sample in the AI Library, take `facedetect` as an example, execute the following command.
```
$cd ~/Vitis-AI/Vitis-AI-Library/overview/samples/facedetect
$bash -x build.sh
```	

7. To modify the library source code, view and modify them under `~/Vitis-AI/Vitis-AI-Library`.
	Before compiling the AI libraries, please confirm the compiled output path. The default output path is : `$HOME/build`.
	If you want to change the default output path, please modify the `build_dir_default` in cmake.sh. 
	Execute the following command to build the libraries all at once.
```
$cd ~/Vitis-AI/Vitis-AI-Library
$./cmake.sh --clean --cmake-options='-DCMAKE_NO_SYSTEM_FROM_IMPORTED=on' 
```

### Setting Up the Target

1. Installing a Board Image.
	* Download the SD card system image files from the following links:  
	
		[ZCU102](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-zcu102-dpu-v2020.1-v1.img.gz)  
	
		[ZCU104](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-zcu104-dpu-v2020.1-v1.img.gz)  
	
      	Note: The version of the board image should be 2020.1 or above.
	* Use Win32DiskImager (free opensource software) to burn the image file onto the SD card.
	* Insert the SD card with the image into the destination board.
	* Plug in the power and boot the board using the serial port to operate on the system.
	* Set up the IP information of the board using the serial port.
	You can now operate on the board using SSH.

2. Installing AI Model Package   
	* Download [ZCU102 AI Model](https://www.xilinx.com/bin/public/openDownload?filename=xilinx_model_zoo_zcu102-1.2.0-1.aarch64.rpm)  
	
		You can also download [ZCU104 AI Model](https://www.xilinx.com/bin/public/openDownload?filename=xilinx_model_zoo_zcu104-1.2.0-1.aarch64.rpm) if you use ZCU104 
	
	* Copy the downloaded file to the board using scp with the following command.
	```
	  $scp xilinx_model_zoo_zcu102-1.2.0-1.aarch64.rpm root@IP_OF_BOARD:~/
	```
	* Log in to the board (usong ssh or serial port) and install the model package.
	* Run the following command.
	```
	  #rpm -ivh xilinx_model_zoo_zcu102-1.2.0-1.aarch64.rpm
	```

3. Installing AI Library Package
	* Download the [Vitis AI Runtime 1.2](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-1.2.0.tar.gz).  

	* Download the [demo video files](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_library_r1.2_video.tar.gz) and untar into the corresponding directories.  
	
	* Download the [demo image files](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_library_r1.2_images.tar.gz) and untar into the corresponding directories.  
	
	* Untar the runtime packet and copy the following folder to the board using scp.
	```
	$tar -xzvf vitis-ai-runtime-1.2.0.tar.gz
	$scp -r vitis-ai-runtime-1.2.0/aarch64/centos root@IP_OF_BOARD:~/
	```
	* Log in to the board using ssh. You can also use the serial port to login.
	* Run the `board_set_up.sh` script. You can also download the `board_set_up.sh` from [here](http://10.176.178.31/mtf/board_set_up.sh).
	```
	#board_set_up.sh
	```
	* For ZCU104, enable the power patch.
	```
	#irps5401
	```
	* For ZCU102, reduce the DPU frequency to 93%.
	```
	#dpu_clk 93
	```
	* Install the Vitis AI Library.
	```
	#cd centos
	#rpm -ivh libunilog-1.2.0-x.aarch64.rpm
	#rpm -ivh libxir-1.2.0-x.aarch64.rpm
	#rpm -ivh libtarget-factory-1.2.0-x.aarch64.rpm
	#rpm -ivh libvart-1.2.0-x.aarch64.rpm
	#rpm -ivh libvitis_ai_library-1.2.0-x.aarch64.rpm
	```
	 	  
### Running Vitis AI Library Examples

1. Copy the sample and demo from host to the target using scp with the following command.
```
[Host]$scp -r ~/Vitis-AI/Vitis-AI-Library/overview root@IP_OF_BOARD:~/
```
2. Copy the image and video packages from host to the target using scp with the following command.
```
[Host]$scp vitis_ai_library_r1.2_images.tar.gz root@IP_OF_BOARD:~/
[Host]$scp vitis_ai_library_r1.2_video.tar.gz root@IP_OF_BOARD:~/
```
3. Untar the image and video packages on the target.
```
#cd ~
#tar -xzvf vitis_ai_library_r1.2_images.tar.gz -C overview
#tar -xzvf vitis_ai_library_r1.2_video.tar.gz -C overview
```
4. Enter the directory of example in target board, take `facedetect` as an example.
```
#cd ~/overview/samples/facedetect
```

5. Run the image test example.
```
#./test_jpeg_facedetect densebox_640_360 sample_facedetect.jpg
```

6. Run the video test example.
```
#./test_video_facedetect densebox_640_360 video_input.webm -t 8
Video_input.mp4: The video file's name for input.The user needs to prepare the videofile.
-t: <num_of_threads>
```
Note that, for examples with video input, only `webm` and `raw` format are supported by default with the official system image. 
If you want to support video data in other formats, you need to install the relevant packages on the system. 

7. To test the program with a USB camera as input, run the following command:
```
#./test_video_facedetect densebox_640_360 0 -t 8
0: The first USB camera device node. If you have multiple USB camera, the value might be 1,2,3 etc.
-t: <num_of_threads>
```

8. To test the performance of model, run the following command:
```
#./test_performance_facedetect densebox_640_360 test_performance_facedetect.list -t 8 -s 60
-t: <num_of_threads>
-s: <num_of_seconds>
```

## Quick Start For Alveo
### Setting Up the Host

Assume the docker image has been loaded and up running.

1. Place the program, data and other files in the workspace folder. After the docker system starts, you will find them under `/workspace` in the docker system.
Do not put the files in any other path of the docker system. They will be lost after you exit the docker system.

2. Run the `alveo_u50_setup.sh` to automatically set up the host for U50, or manully perform the following steps from 3 to 6. If you run the `alveo_u50_setup.sh` successfully, then skip to step 7.
```
$bash -x alveo_u50_setup.sh
```
3. Download [vitis_ai_runtime_library_r1.2](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-1.2.0.tar.gz) package.
Untar it, find the `libvitis_ai_library_1.2.0-rx_amd64.deb` package and install it to the docker system.
```
$wget https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-1.2.0.tar.gz -O vitis-ai-runtime-1.2.0.tar.gz
$tar -xzvf vitis-ai-runtime-1.2.0.tar.gz
$cd vitis-ai-runtime-1.2.0/X86_64/ubuntu
$sudo dpkg -i libvitis_ai_library_1.2.0-r4_amd64.deb
```
4. Select the model for your platform, download the model packet and install it. Take `U50` as an example. 
```
$wget https://www.xilinx.com/bin/public/openDownload?filename=xilinx_model_zoo_u50_1.2.0_amd64.deb -O xilinx_model_zoo_u50_1.2.0_amd64.deb
$sudo dpkg -i xilinx_model_zoo_u50_1.2.0_amd64.deb
```
<details>
 <summary><b>Click here to download models for different alveo cards </b></summary>
 
| No\. | Alveo              | Download Link                                                      |
| :--- | :----------------------- | :----------------------------------------------------------- |
| 1    | U50             | [xilinx_model_zoo_u50_6e](https://www.xilinx.com/bin/public/openDownload?filename=xilinx_model_zoo_u50_1.2.0_amd64.deb)                              |
| 2    | U50lv           | [xilinx_model_zoo_u50lv_9e](https://www.xilinx.com/bin/public/openDownload?filename=xilinx_model_zoo_u280_u50lv9e_1.2.0_amd64.deb)                       |
| 3    | U50lv           | [xilinx_model_zoo_u50lv_10e](https://www.xilinx.com/bin/public/openDownload?filename=xilinx_model_zoo_u50lv10e_1.2.0_amd64.deb)                  |
| 4    | U280            | [xilinx_model_zoo_u280_14e](https://www.xilinx.com/bin/public/openDownload?filename=xilinx_model_zoo_u280_u50lv9e_1.2.0_amd64.deb)            |

</details>

5. Download the [Alveo_xclbin](https://www.xilinx.com/bin/public/openDownload?filename=alveo_xclbin-1.2.0.tar.gz). Untar it, choose the alveo card and install it. Take `U50` as an example.
```
$wget https://www.xilinx.com/bin/public/openDownload?filename=alveo_xclbin-1.2.0.tar.gz -O alveo_xclbin-1.2.0.tar.gz
$tar -xzvf alveo_xclbin-1.2.0.tar.gz
$cd alveo_xclbin-1.2.0/U50/6E300M
$sudo cp dpu.xclbin hbm_address_assignment.txt /usr/lib
```
6. Enable environment variable and export the library path.
```
$export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:/usr/lib:/usr/lib/x86_64-linux-gnu:/opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/
```
7. To compile the demo in the AI Library, take `yolov3` as an example.
```
$cd /workspace/Vitis-AI-Library/overview/demo/yolov3
$bash -x build.sh
```	
8. To compile the AI Library sample, take `classification` as an example, execute the following command.
```
$cd /workspace/Vitis-AI-Library/overview/samples/classification
$bash -x build.sh
```	

9. To modify the library source code, view and modify them under `/workspace/Vitis-AI/Vitis-AI-Library`.
	Before compiling the AI libraries, please confirm the compiled output path. The default output path is : `$HOME/build`.
	If you want to change the default output path, please modify the `build_dir_default` in cmake.sh. 
	Execute the following command to build the libraries all at once.
```
$cd /workspace/Vitis-AI-Library
$./cmake.sh --clean --cmake-options='-DCMAKE_NO_SYSTEM_FROM_IMPORTED=on' 
```

### Running Vitis AI Library Examples
1. Download the [vitis_ai_library_r1.2_images.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_library_r1.2_images.tar.gz) and [vitis_ai_library_r1.2_video.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_library_r1.2_video.tar.gz) packages and untar them.
```
$cd /workspace
$wget https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_library_r1.2_images.tar.gz -O vitis_ai_library_r1.2_images.tar.gz
$wget https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_library_r1.2_video.tar.gz -O vitis_ai_library_r1.2_video.tar.gz
$tar -xzvf vitis_ai_library_r1.2_images.tar.gz -C Vitis-ai/Vitis-AI-Library/overview
$tar -xzvf vitis_ai_library_r1.2_video.tar.gz -C Vitis-ai/Vitis-AI-Library/overview
```
2. Enter the directory of sample and then compile it. Take `facedetect` as an example.
```
$cd /workspace/vitis-ai/vitis_ai_library/overview/samples/facedetect
$bash -x build.sh
```
3. Run the image test example.
```
$./test_jpeg_facedetect densebox_640_360 sample_facedetect.jpg
```
4. If you want to run the program in batch mode, which means that the DPU processes multiple
images at once to prompt for processing performance, you have to compile the entire Vitis AI
Library according to "Setting Up the Host For Cloud" section. Then the batch program will be generated
under build_dir_default.Enter build_dir_default, take facedetect as an example,
execute the following command.
```
$./test_facedetect_batch densebox_640_360 <img1_url> [<img2_url> ...]
```
5. Run the video test example.
```
#./test_video_facedetect densebox_640_360 video_input.mp4 -t 8
Video_input.mp4: The video file's name for input.The user needs to prepare the videofile.
-t: <num_of_threads>
```
6. To test the performance of model, run the following command:
```
#./test_performance_facedetect densebox_640_360 test_performance_facedetect.list -t 8 -s 60
-t: <num_of_threads>
-s: <num_of_seconds>
```

## Reference
For more information, please refer to [vitis-ai-library-user-guide](https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_1/ug1354-xilinx-ai-sdk.pdf) and [vitis-ai-library-programming-guide](https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_1/ug1355-xilinx-ai-sdk-programming-guide.pdf).
	
