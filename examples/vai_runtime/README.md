<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

# Vitis AI Runtime Introduction
The Vitis AI Runtime (VART) enables applications to use the unified high-level runtime API for both data center and embedded. Therefore, making cloud-to-edge deployments seamless and efficient.
The Vitis AI Runtime API features are:
* Asynchronous submission of jobs to the accelerator
* Asynchronous collection of jobs from the accelerator
* C++ and Python implementations
* Support for multi-threading and multi-process execution

For embedded users, click 
[Quick Start For Embedded](#quick-start-for-embedded) to get started quickly. 

For cloud users, click 
[Quick Start For Data Center](#quick-start-for-data-center) to get started quickly.

Vitis AI Runtime directory structure introduction
--------------------------------------------------

```
vai_runtime
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

## Quick Start For Embedded
### Setting Up the Host
For `VEK280`, follow [Setting Up the Host](https://xilinx.github.io/Vitis-AI/3.5/html/docs/quickstart/vek280.html#setup-the-host)) to set up the host for edge.

### Setting Up the Target
For `VEK280`, follow [Setting Up the Target](https://xilinx.github.io/Vitis-AI/3.5/html/docs/quickstart/vek280.html#setup-the-target) to set up the target.
	 	  
### Running Vitis AI Examples

Follow [Running Vitis AI Examples](https://xilinx.github.io/Vitis-AI/3.5/html/docs/quickstart/vek280.html#run-the-vitis-ai-examples) to run Vitis AI examples.

Note: When you update from VAI1.3 to VAI2.0, VAI2.5, VAI3.0 or VAI3.5, refer to the following to modify your compilation options.
1. For Petalinux 2021.1 and above, it uses OpenCV4, and for Petalinux 2020.2, it uses OpenCV3. So set the `OPENCV_FLAGS` as needed. You can refer to the following.
```
result=0 && pkg-config --list-all | grep opencv4 && result=1
if [ $result -eq 1 ]; then
	OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv4)
else
	OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv)
fi
```

## Quick Start For Data Center
### Setting Up the Host

For `V70` Versal Card, follow [Setup V70 Accelerator Card](https://xilinx.github.io/Vitis-AI/3.5/html/docs/quickstart/v70.html#versal-v70-setup) to set up the host.

### Running Vitis AI Examples
In the docker system, `/workspace/examples/vai_runtime/` is the path for the following example. If you encounter any path errors in running examples, check to see if you follow the steps above to set the host. Then, follow the steps below to download the model and run the sample, take `resnet50` as an example.

1. Download the [vitis_ai_runtime_r3.5.x_image_video.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_runtime_r3.5.0_image_video.tar.gz) package and unzip it.
	```
	cd /workspace/examples
	wget https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_runtime_r3.5.0_image_video.tar.gz -O vitis_ai_runtime_r3.5.0_image_video.tar.gz
	tar -xzvf vitis_ai_runtime_r3.5.0_image_video.tar.gz -C vai_runtime
	```
2. Download the model.

	* If the `/usr/share/vitis_ai_library/models` folder does not exist, create it first.
	```
	  sudo mkdir /usr/share/vitis_ai_library/models
	```

	* For DPUCV2DX8G_v70 DPU IP, install the model package as follows.
	```
	  wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-v70-DPUCV2DX8G-r3.5.0.tar.gz -O resnet50-v70-DPUCV2DX8G-r3.5.0.tar.gz
	  tar -xzvf resnet50-v70-DPUCV2DX8G-r3.5.0.tar.gz
	  sudo cp resnet50 /usr/share/vitis_ai_library/models -r
	```

3. Compile the example.
	```
	cd /workspace/examples/vai_runtime/resnet50
	bash -x build.sh
	```
4. Run the example.
	```
	./resnet50 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel
	```
	**Note that different Versal cards DPU IP correspond to different model files, which cannot be used alternately.** 


 <summary><b>Launching Commands for VART Samples on V70 </b></summary>
 
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


