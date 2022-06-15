<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Vitis AI Runtime v2.5</h1>
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
[Quick Start For Cloud](#quick-start-for-cloud) to get started quickly.

Vitis AI Runtime directory structure introduction
--------------------------------------------------

```
VART
├── README.md
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
For `MPSOC`, follow [Setting Up the Host](../../setup/mpsoc#step1-setup-cross-compiler) to set up the host for edge.  
For `VCK190`, follow [Setting Up the Host](../../setup/vck190#step1-setup-cross-compiler) to set up the host for edge.

### Setting Up the Target
For `MPSOC`, follow [Setting Up the Target](../../setup/mpsoc/README.md#step2-setup-the-target) to set up the target.  
For `VCK190`, follow [Setting Up the Target](../../setup/vck190/README.md#step2-setup-the-target) to set up the target.
	  
### Running Vitis AI Examples

Follow [Running Vitis AI Examples](../../setup/mpsoc/README.md#step3-run-the-vitis-ai-examples) to run Vitis AI examples.

Note: When you update from VAI1.3 to VAI2.0 and VAI2.5, refer to the following to modify your compilation options.
1. For Petalinux 2021.1 and above, it uses OpenCV4, and for Petalinux 2020.2, it uses OpenCV3. So set the `OPENCV_FLAGS` as needed. You can refer to the following.
```
result=0 && pkg-config --list-all | grep opencv4 && result=1
if [ $result -eq 1 ]; then
	OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv4)
else
	OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv)
fi
```

## Quick Start For Cloud
### Setting Up the Host

For demonstration purposes, we provide the following pre-compiled DPU IP with VART Sample support. You can choose one of them according to your own Accelerator Card.

| No\. | Accelerator Card | DPU IP |
| ---- | ---- | ----   |
| 1 | U50LV        | DPUCAHX8H         |
| 2 | U50LV        | DPUCAHX8H-DWC     |
| 3 | U55C         | DPUCAHX8H-DWC     |
| 4 | VCK5000-PROD | DPUCVDX8H_4pe_miscdwc |
| 5 | VCK5000-PROD | DPUCVDX8H_6pe_dwc     |
| 6 | VCK5000-PROD | DPUCVDX8H_6pe_misc    |
| 7 | VCK5000-PROD | DPUCVDX8H_8pe_normal  |

For `U50LV` and `U55C` Alveo Card, follow [Setup Alveo Accelerator Card](../../setup/alveo/README.md) to set up the host.  
For `VCK5000-PROD` Versal Card, follow [Setup VCK5000 Accelerator Card](../../setup/vck5000/README.md) to set up the host.

### Running Vitis AI Examples
In the docker system, `/workspace/examples/VART/` is the path for the following example. If you encounter any path errors in running examples, check to see if you follow the steps above to set the host. Then, follow the steps below to download the model and run the sample.

1. Download the [vitis_ai_runtime_r2.5.x_image_video.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_runtime_r2.5.0_image_video.tar.gz) package and unzip it.
	```
	cd /workspace/examples
	wget https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_runtime_r2.5.0_image_video.tar.gz -O vitis_ai_runtime_r2.5.0_image_video.tar.gz
	tar -xzvf vitis_ai_runtime_r2.5.0_image_video.tar.gz -C VART
	```
2. Download the model.  	
	For each model, there will be a yaml file which is used for describe all the details about the model. 
	In the yaml, you will find the model's download links for different platforms. Please choose the corresponding model and download it. Click [Xilinx AI Model Zoo](../../model_zoo/model-list) to view all the models. Take [resnet50  yaml file](../../model_zoo/model-list/cf_resnet50_imagenet_224_224_7.7G_2.5/model.yaml) as an example.

	* If the `/usr/share/vitis_ai_library/models` folder does not exist, create it first.
	```
	  sudo mkdir /usr/share/vitis_ai_library/models
	```

	* For DPUCAHX8H DPU IP of U50LV card, install the model package as follows.
	```
	  wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-u50lv-DPUCAHX8H-r2.5.0.tar.gz -O resnet50-u50lv-DPUCAHX8H-r2.5.0.tar.gz
	  tar -xzvf resnet50-u50lv-DPUCAHX8H-r2.5.0.tar.gz
	  sudo cp resnet50 /usr/share/vitis_ai_library/models -r
	```

	* For DPUCAHX8H-DWC DPU IP of U50LV and U55C card, install the model package as follows.
	```
	  wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-u55c-u50lv-DPUCAHX8H-DWC-r2.5.0.tar.gz -O resnet50-u55c-u50lv-DPUCAHX8H-DWC-r2.5.0.tar.gz
	  tar -xzvf resnet50-u55c-u50lv-DPUCAHX8H-DWC-r2.5.0.tar.gz
	  sudo cp resnet50 /usr/share/vitis_ai_library/models -r
	```

3. Compile the sample, take `resnet50` as an example.
	```
	cd /workspace/examples/VART/resnet50
	bash -x build.sh
	```
4. Run the example, take `U50LV` platform as an example.
	```
	./resnet50 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel
	```
	**Note that different alveo cards correspond to different model files, which cannot be used alternately.** 


 <summary><b>Launching Commands for VART Samples on U50lv/U55C/VCK5000-PROD </b></summary>
 
| No\. | Example Name             | Command                                                   |
| :--- | :----------------------- | :-------------------------------------------------------- |
| 1    | resnet50                 | ./resnet50 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel                            |
| 2    | resnet50_pt              | ./resnet50_pt /usr/share/vitis_ai_library/models/resnet50_pt/resnet50_pt.xmodel ../images/001.jpg |
| 3    | resnet50_ext             | ./resnet50_ext /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel ../images/001.jpg                           |
| 4    | resnet50_mt_py           | /usr/bin/python3 resnet50.py 1 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel          |
| 5    | inception_v1_mt_py       | /usr/bin/python3 inception_v1.py 1 /usr/share/vitis_ai_library/models/inception_v1_tf/inception_v1_tf.xmodel      |
| 6    | pose_detection           | ./pose_detection video/pose.webm /usr/share/vitis_ai_library/models/sp_net/sp_net.xmodel /usr/share/vitis_ai_library/models/ssd_pedestrian_pruned_0_97/ssd_pedestrian_pruned_0_97.xmodel         |
| 7    | video_analysis           | ./video_analysis video/structure.webm /usr/share/vitis_ai_library/models/ssd_traffic_pruned_0_9/ssd_traffic_pruned_0_9.xmodel    |
| 8    | adas_detection           | ./adas_detection video/adas.webm /usr/share/vitis_ai_library/models/yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel         |
| 9    | segmentation             | ./segmentation video/traffic.webm /usr/share/vitis_ai_library/models/fpn/fpn.xmodel        |
| 10   | squeezenet_pytorch       | ./squeezenet_pytorch /usr/share/vitis_ai_library/models/squeezenet_pt/squeezenet_pt.xmodel        |


