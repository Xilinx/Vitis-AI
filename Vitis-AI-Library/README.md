<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Vitis AI Library v1.1</h1>
    </td>
 </tr>
 </table>

# Introduction
The Vitis AI Library is a set of high-level libraries and APIs built for efficient AI inference with Deep-Learning Processor Unit (DPU). It is built based on the Vitis AI Runtime with Unified APIs, and it fully supports XRT 2019.2.

The Vitis AI Library provides an easy-to-use and unified interface by encapsulating many efficient and high-quality neural networks. This simplifies the use of deep-learning neural networks, even for users without knowledge of deep-learning or FPGAs. The Vitis AI Library allows users to focus more on the development of their applications, rather than the underlying hardware.


<p align="center">
  <img src="ai_library_diagram.png" >
</p>

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
│   ├── README.md
│   ├── samples
│   ├── src
│   └── test
├── cmake
│   ├── FindUnilog.cmake
│   ├── FindXir.cmake
│   ├── XilinxCommon.cmake
│   ├── XilinxDpu.cmake
│   ├── xilinx_version.c.in
│   └── XilinxVersion.cmake
├── CMakeLists.txt
├── cmake.sh
├── Copyright.txt
├── dpu_task
│   ├── CMakeLists.txt
│   ├── include
│   ├── readme.md
│   ├── README.md
│   ├── src
│   ├── test
│   └── util
├── facedetect
│   ├── CMakeLists.txt
│   ├── include
│   ├── README.md
│   ├── src
│   └── test
├── facelandmark
│   ├── CMakeLists.txt
│   ├── include
│   ├── README.md
│   ├── src
│   └── test
├── lanedetect
│   ├── CMakeLists.txt
│   ├── include
│   ├── README.md
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
├── model_config
│   ├── CMakeLists.txt
│   ├── include
│   ├── src
│   └── test
├── multitask
│   ├── CMakeLists.txt
│   ├── include
│   ├── README.md
│   ├── src
│   └── test
├── openpose
│   ├── CMakeLists.txt
│   ├── include
│   ├── README.md
│   ├── src
│   └── test
├── overview
│   ├── cmake
│   ├── CMakeLists.txt
│   ├── demo						#AI Library demo
│   │   ├── classification
│   │   ├── seg_and_pose_detect
│   │   ├── segs_and_roadline_detect
│   │   └── yolov3
│   └── samples						#AI Library samples
│       ├── classification
│       ├── facedetect
│       ├── facelandmark
│       ├── lanedetect
│       ├── multitask
│       ├── openpose
│       ├── posedetect
│       ├── refinedet
│       ├── reid
│       ├── segmentation
│       ├── ssd
│       ├── tfssd
│       ├── yolov2
│       └── yolov3
├── posedetect
│   ├── CMakeLists.txt
│   ├── include
│   ├── README.md
│   ├── src
│   └── test
├── README.md
├── refinedet
│   ├── CMakeLists.txt
│   ├── include
│   ├── README.md
│   ├── src
│   └── test
├── reid
│   ├── CMakeLists.txt
│   ├── include
│   ├── README.md
│   ├── src
│   └── test
├── segmentation
│   ├── CMakeLists.txt
│   ├── include
│   ├── README.md
│   ├── src
│   └── test
├── ssd
│   ├── CMakeLists.txt
│   ├── include
│   ├── README.md
│   ├── src
│   └── test
├── tfssd
│   ├── CMakeLists.txt
│   ├── include
│   ├── README.md
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
│   ├── CMakeLists.txt.bac
│   ├── cscope.files
│   ├── include
│   └── src
├── yolov2
│   ├── CMakeLists.txt
│   ├── include
│   ├── README.md
│   ├── src
│   └── test
└── yolov3
    ├── CMakeLists.txt
    ├── include
    ├── README.md
    ├── src
    └── test

```

## Quick Start
### Setting Up the Host For Edge
1. Download the [sdk.sh](https://www.xilinx.com/bin/public/openDownload?filename=sdk.sh)

2. Install the cross-compilation system environment, follow the prompts to install. 
```
$./sdk.sh
```
Note that the `~/petalinux_sdk` path is recommended for the installation. Regardless of the path you choose for the installation, make sure the path has read-write permissions. 
Here we install it under `~/petalinux_sdk`.

3. When the installation is complete, follow the prompts and execute the following command.
```
$. ~/petalinux_sdk/environment-setup-aarch64-xilinx-linux
```
Note that if you close the current terminal, you need to re-execute the above instructions in the new terminal interface.

4. Download the [vitis_ai_2019.2-r1.1.0.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_2019.2-r1.1.0.tar.gz) and install it to the petalinux system.
```
$tar -xzvf vitis_ai_2019.2-r1.1.0.tar.gz -C ~/petalinux_sdk/sysroots/aarch64-xilinx-linux
```
5. Update the glog to v0.4.0
	* Download the glog package and untar it.
		```
		$cd ~
		$curl -Lo glog-v0.4.0.tar.gz https://github.com/google/glog/archive/v0.4.0.tar.gz
		$tar -zxvf glog-v0.4.0.tar.gz
		$cd glog-0.4.0
		```
	* Build it and install it to the PetaLinux system.
		```
		$mkdir build_for_petalinux
		$cd build_for_petalinux
		$unset LD_LIBRARY_PATH; source ~/petalinux_sdk/environment-setup-aarch64-xilinx-linux
		$cmake -DCPACK_GENERATOR=TGZ -DBUILD_SHARED_LIBS=on -DCMAKE_INSTALL_PREFIX=$OECORE_TARGET_SYSROOT/usr ..
		$make && make install
		$make package
		```

6. Cross compile the demo in the AI Library, using yolov3 as example.
```
$cd ~/Vitis-AI/Vitis-AI-Library/overview/demo/yolov3
$bash -x build.sh
```	

7. To compile the library sample in the AI Library, take classification for example, execute the following command.
```
$cd ~/Vitis-AI/Vitis-AI-Library/overview/samples/classification
$bash -x build.sh
```	

8. To modify the library source code, view and modify them under `~/Vitis-AI/Vitis-AI-Library`.
	Before compiling the AI libraries, please confirm the compiled output path. The default output path is : `$HOME/build`.
	If you want to change the default output path, please modify the `build_dir_default` in cmake.sh. 
	Execute the following command to build the libraries all at once.
```
$cd ~/Vitis-AI/Vitis-AI-Library
$./cmake.sh --clean --cmake-options='-DCMAKE_NO_SYSTEM_FROM_IMPORTED=on' 
```

### Setting Up the Host For Alveo

Assume the docker image has been loaded and up running.

1. Place the program, data and other files in the workspace folder. After the docker system starts, you will find them under `/workspace` in the docker system.
Do not put the files in any other path of the docker system. They will be lost after you exit the docker system.

2. Download [vitis_ai_runtime_library_r1.1](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-1.1.0.tar.gz) package.
Untar it, find the `libvitis_ai_library-1.1.0-Linux-build<xx>.deb` package and install it to the docker system.
```
$sudo dpkg -i libvitis_ai_library-1.1.0-Linux.deb
```
5. Download [U50 Model](https://www.xilinx.com/bin/public/openDownload?filename=xilinx_model_zoo-1.1.0-Linux.deb) packet and install it.
```
$sudo dpkg -i xilinx_model_zoo-1.1.0-Linux.deb
```
6. Download the [U50_xclbin](https://www.xilinx.com/bin/public/openDownload?filename=U50_xclbin.tar.gz) and install them.
```
$sudo cp dpu.xclbin hbm_address_assignment.txt /usr/lib
```
7. Enable environment variable and export the library path.
```
$export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:/usr/lib:/usr/lib/x86_64-linux-gnu:/opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/
```

8. To compile the demo in the AI Library, take `yolov3` as an example.
```
$cd /workspace/Vitis-AI/Vitis-AI-Library/overview/demo/yolov3
$bash -x build.sh
```	
9. To compile the AL Library sample, take `classification` as an example, execute the following command.
```
$cd /workspace/Vitis-AI-Library/overview/samples/classification
$bash -x build.sh
```	

10. To modify the library source code, view and modify them under `/workspace/Vitis-AI/Vitis-AI-Library`.
	Before compiling the AI libraries, please confirm the compiled output path. The default output path is : `$HOME/build`.
	If you want to change the default output path, please modify the `build_dir_default` in cmake.sh. 
	Execute the following command to build the libraries all at once.
```
$cd /workspace/Vitis-AI-Library
$./cmake.sh --clean --cmake-options='-DCMAKE_NO_SYSTEM_FROM_IMPORTED=on' 
```


### Setting Up the Target

1. Installing a Board Image.
	* Download the SD card system image files from the following links:  
	
		[ZCU102](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-zcu102-dpu-v2019.2-v2.img.gz)  
	
		[ZCU104](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-zcu104-dpu-v2019.2-v2.img.gz)  
	
      	Note: The version of the board image should be 2019.2 or above.
	* Use Win32DiskImager (free opensource software) to burn the image file onto the SD card.
	* Insert the SD card with the image into the destination board.
	* Plug in the power and boot the board using the serial port to operate on the system.
	* Set up the IP information of the board using the serial port.
	You can now operate on the board using SSH.

2. Installing AI Model Package   
	* Download [ZCU102 AI Model](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_model_ZCU102_2019.2-r1.1.0.deb)  
	
		You can also download [ZCU104 AI Model](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_model_ZCU104_2019.2-r1.1.0.deb) if you use ZCU104 
	
	* Copy the downloaded file to the board using scp with the following command.
	```
	  $scp vitis_ai_model_ZCU102_2019.2-r1.1.0.deb root@IP_OF_BOARD:~/
	```
	* Log in to the board (usong ssh or serial port) and install the model package.
	* Run the following command.
	```
	  #dpkg -i vitis_ai_model_ZCU102_2019.2-r1.1.0.deb
	```

3. Installing AI Library Package
	* Download the [Vitis AI Runtime 1.1](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-1.1.0.tar.gz). 
Untar it and find the `libvitis_ai_library-1.1.0-Linux-build46.deb` package in /vitis-ai-runtime-1.1.0/Vitis-AI-Library/aarch64 directory. 

	* Download the [demo video files](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_library_r1.1_video.tar.gz) and untar into the corresponding directories.  
	
	* Download the [demo image files](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_library_r1.1_images.tar.gz)  and untar into the corresponding directories.  
	
	* Copy the downloaded file to the board using scp with the following command.
	```
	  $scp libvitis_ai_library-1.1.0-Linux-build46.deb root@IP_OF_BOARD:~/
	```
	* Copy the `glog-0.4.0-Linux.tar.gz` from host to board with the following command. 
	```
	$cd ~/glog-0.4.0/build_for_petalinux
	$scp glog-0.4.0-Linux.tar.gz root@IP_OF_BOARD:~/
	```
	* Log in to the board using ssh. You can also use the serial port to login.
	* Update the glog to v0.4.0.
	```
	#tar -xzvf glog-0.4.0-Linux.tar.gz --strip-components=1 -C /usr
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
	#dpkg -i libvitis_ai_library-1.1.0-Linux-build46.deb
	```
	 
	  
### Running Vitis AI Library Examples (For Edge)

1. Copy the sample and demo from host to the target using scp with the following command.
```
$scp -r ~/Vitis-AI/Vitis-AI-Library/overview root@IP_OF_BOARD:~/
```
2. Copy the image and video packages from host to the target using scp with the following command.
```
$scp vitis_ai_library_r1.1_images.tar.gz root@IP_OF_BOARD:~/
$scp vitis_ai_library_r1.1_video.tar.gz root@IP_OF_BOARD:~/
```
3. Untar the image and video packages on the target.
```
#cd ~
#tar -xzvf vitis_ai_library_r1.1_images.tar.gz -C overview
#tar -xzvf vitis_ai_library_r1.1_video.tar.gz -C overview
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
#./test_video_facedetect densebox_640_360 video_input.mp4 -t 8
Video_input.mp4: The video file's name for input.The user needs to prepare the videofile.
-t: <num_of_threads>
```

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

9. To check the version of Vitis AI Library, run the following command:
```
#vitis_ai
```	

### Running Vitis AI Library Examples (For Cloud)
1. Download the [vitis_ai_library_r1.1_images](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_library_r1.1_images.tar.gz) and [vitis_ai_library_r1.1_video.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_library_r1.1_video.tar.gz) packages.
2. Untar the image and video packages.
```
#cd /workspace
#tar -xzvf vitis_ai_library_r1.1_images.tar.gz -C Vitis-ai/Vitis-AI-Library/overview
#tar -xzvf vitis_ai_library_r1.1_video.tar.gz -C Vitis-ai/Vitis-AI-Library/overview
```
3. Enter the directory of sample and then compile it. Take `facedetect` as an example.
```
$cd /workspace/vitis-ai/vitis_ai_library/overview/samples/facedetect
$bash -x build.sh
```
4. Run the image test example.
```
$./test_jpeg_facedetect densebox_640_360 sample_facedetect.jpg
```
5. If you want to run the program in batch mode, which means that the DPU processes multiple
images at once to prompt for processing performance, you have to compile the entire Vitis AI
Library according to "Setting Up the Host For Cloud" section. Then the batch program will be generated
under build_dir_default.Enter build_dir_default, take facedetect as an example,
execute the following command.
```
$./test_facedetect_batch densebox_640_360 <img1_url> [<img2_url> ...]
```
6. Run the video test example.
```
#./test_video_facedetect densebox_640_360 video_input.mp4 -t 8
Video_input.mp4: The video file's name for input.The user needs to prepare the videofile.
-t: <num_of_threads>
```
7. To test the performance of model, run the following command:
```
#./test_performance_facedetect densebox_640_360 test_performance_facedetect.list -t 8 -s 60
-t: <num_of_threads>
-s: <num_of_seconds>
```
8. To check the version of Vitis AI Library, run the following command:
```
#vitis_ai
```	
## Reference
For more information, please refer to [vitis-ai-library-user-guide](https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_1/ug1354-xilinx-ai-sdk.pdf) and [vitis-ai-library-programming-guide](https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_1/ug1355-xilinx-ai-sdk-programming-guide.pdf).
	
