<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Vitis AI Runtime v1.1</h1>
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
└── samples
    ├── adas_detection
    │   ├── build.sh
    │   ├── model_dir_for_U50
    │   ├── model_dir_for_zcu102
    │   ├── model_dir_for_zcu104
    │   └── src
    ├── common
    │   ├── common.cpp
    │   └── common.h
    ├── inception_v1_mt_py
    │   ├── inception_v1.py
    │   ├── input_fn.py
    │   ├── model_dir_for_U50
    │   ├── model_dir_for_zcu102
    │   └── model_dir_for_zcu104
    ├── pose_detection
    │   ├── build.sh
    │   ├── model_dir_for_U50
    │   ├── model_dir_for_zcu102
    │   ├── model_dir_for_zcu104
    │   └── src
    ├── resnet50
    │   ├── build.sh
    │   ├── model_dir_for_U50
    │   ├── model_dir_for_zcu102
    │   ├── model_dir_for_zcu104
    │   ├── src
    │   └── words.txt
    ├── resnet50_mt_py
    │   ├── input_fn.py
    │   ├── input.py
    │   ├── model_dir_for_U50
    │   ├── model_dir_for_zcu102
    │   ├── model_dir_for_zcu104
    │   └── resnet50.py
    ├── segmentation
    │   ├── build.sh
    │   ├── model_dir_for_U50
    │   ├── model_dir_for_zcu102
    │   ├── model_dir_for_zcu104
    │   └── src
    └── video_analysis
        ├── build.sh
        ├── model_dir_for_U50
        ├── model_dir_for_zcu102
        ├── model_dir_for_zcu104
        └── src


```

## Quick Start For Edge
### Setting Up the Host
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

6. Cross compile the sample, take resnet50 as an example.
```
$cd ~/vitis-ai/VART/samples/resnet50
$bash –x build.sh
```	
If the compilation process does not report any error and the executable file `resnet50` is generated, the host environment is installed correctly.

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

2. Installing Vitis AI Runtime Package   
	* Download [vitis_ai_runtime_library_r1.1](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-1.1.2.tar.gz)  
	
	* Connect to the board with SSH: $ssh root@IP_OF_BOARD. The password is `root`.
	
	* Untar the package and copy the following files to the board using scp.
	```
	$scp <path_to_untar'd_runtime_library>/unilog/aarch64/libunilog-1.1.0-Linux-build<xx>.deb root@IP_OF_BOARD:~/
	$scp <path_to_untar'd_runtime_library>/XIR/aarch64/libxir-1.1.0-Linux-build<xx>.deb root@IP_OF_BOARD:~/
	$scp <path_to_untar'd_runtime_library>/VART/aarch64/libvart-1.1.0-Linux-build<xx>.deb root@IP_OF_BOARD:~/
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
	* Install the Vitis AI Runtime. Execute the following command in order.
	```
	#dpkg –i --force-all libunilog-1.1.0-Linux-build<xx>.deb
	#dpkg –i libxir-1.1.0-Linux-build<xx>.deb
	#dpkg –i libvart-1.1.0-Linux-build<xx>.deb
	```
	* For ZCU104, enable the power patch.
	```
	#irps5401
	```	 
	  
### Running Vitis AI Examples

1. Download the samples from host to the target using scp with the following command.
	```
	[Host]$scp -r ~/Vitis-AI/VART root@[IP_OF_BOARD]:~/
	```
2. Download the [vitis_ai_runtime_r1.1_image_video.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_runtime_r1.1_image_video.tar.gz) from host to the target using scp with the following command.
	```
	[Host]$scp vitis_ai_runtime_r1.1_image_video.tar.gz root@[IP_OF_BOARD]:~/
	```
3. Unzip the `vitis_ai_runtime_r1.1_image_video.tar.gz` package on the target.
	```
	#cd ~
	#tar -xzvf vitis_ai_runtime_r1.1_image_video.tar.gz -C VART
	```
4. Enter the directory of samples in the target board. Take resnet50 as an example.
	```
	#cd ~/VART/samples/resnet50
	```
5. Run the example.
	```
	#./resnet50 model_dir_for_zcu102
	# For ZCU104, execute the following command.
	#./resnet50 model_dir_for_zcu104
	```
	Note that if the above executable program does not exist, run the following command to compile and generate the corresponding executable program.
	```
	#bash –x build.sh
	```
<details>
 <summary><b>Click here to view Launching Commands for Vitis AI Samples on ZCU102 </b></summary>
 
| No\. | Example Name             | Command                                                      |
| :--- | :----------------------- | :----------------------------------------------------------- |
| 1    | resnet50                 | ./resnet50 model_dir_for_zcu102                              |
| 2    | resnet50_mt_py           | python3 resnet50.py 1 model_dir_for_zcu102                   |
| 3    | inception_v1_mt_py       | python3 inception_v1.py 1 model_dir_for_zcu102               |
| 4    | pose_detection           | ./pose_detection video/pose.mp4 model_dir_for_zcu102         |
| 5    | video_analysis           | ./video_analysis video/structure.mp4 model_dir_for_zcu102    |
| 6    | adas_detection           | ./adas_detection video/adas.avi model_dir_for_zcu102         |
| 7    | segmentation             | ./segmentation video/traffic.mp4 model_dir_for_zcu102        |

</details>

## Quick Start For Alveo
### Setting Up the Host

Assume the docker image has been loaded and up running.

1. Download the [U50_xclbin](https://www.xilinx.com/bin/public/openDownload?filename=U50_xclbin-v2.tar.gz) and install them.
```
$wget https://www.xilinx.com/bin/public/openDownload?filename=U50_xclbin-v2.tar.gz -O U50_xclbin-v2.tar.gz
$tar -xzvf U50_xclbin-v2.tar.gz
$cd U50_xclbin/6E250M
$sudo cp dpu.xclbin hbm_address_assignment.txt /usr/lib
```
2. Enable environment variable and export the library path.
```
$export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:/usr/lib:/usr/lib/x86_64-linux-gnu:/opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/
```
3. Download [vitis_ai_runtime_library_r1.1](https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-1.1.2.tar.gz) , untar the packet and install the VART runtime.
```
$wget https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-1.1.2.tar.gz -O vitis-ai-runtime-1.1.2.tar.gz
$tar -xzvf vitis-ai-runtime-1.1.2.tar.gz
$cd vitis-ai-runtime-1.1.2/VART/X86_64/
$sudo dpkg –i libvart-1.1.0-Linux-build<xx>.deb
```

### Running Vitis AI Examples
1. Download the [vitis_ai_runtime_r1.1_image_video.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_runtime_r1.1_image_video.tar.gz) package and unzip it.
	```
	$cd /workspace
	$wget https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_runtime_r1.1_image_video.tar.gz -O vitis_ai_runtime_r1.1_image_video.tar.gz
	$tar -xzvf vitis_ai_runtime_r1.1_image_video.tar.gz -C VART
	```
2. Compile the sample, take resnet50 as example.
	```
	$cd /workspace/VART/samples/resnet50
	$bash –x build.sh
	```
3. Run the example.
	```
	$./resnet50 model_dir_for_U50
	```

<details>
 <summary><b>Click here to view Launching Commands for Vitis AI Samples on U50 </b></summary>
 
| No\. | Example Name             | Command                                                   |
| :--- | :----------------------- | :-------------------------------------------------------- |
| 1    | resnet50                 | ./resnet50 model_dir_for_U50                              |
| 2    | resnet50_mt_py           | /usr/bin/python3 resnet50.py 1 model_dir_for_U50          |
| 3    | inception_v1_mt_py       | /usr/bin/python3 inception_v1.py 1 model_dir_for_U50      |
| 4    | pose_detection           | ./pose_detection video/pose.mp4 model_dir_for_U50         |
| 5    | video_analysis           | ./video_analysis video/structure.mp4 model_dir_for_U50    |
| 6    | adas_detection           | ./adas_detection video/adas.avi model_dir_for_U50         |
| 7    | segmentation             | ./segmentation video/traffic.mp4 model_dir_for_U50        |

</details>
