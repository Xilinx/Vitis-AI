<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Vitis AI v1.3</h1>
    </td>
 </tr>
 </table>

# Introduction
There are four demos as shown in the following.
* classification
* seg_and_pose_detect
* segs_and_roadline_detect
* yolov3

All the demos can run on either edge or cloud board.

## Demo For Edge
### Setting Up the Host

1. Set up the host environment according to [Setting-up-the-host](../examples/Vitis-AI-Library#setting-up-the-host)

2. Cross compile the demo, take `seg_and_pose_detect` as example.
```
cd Vitis-AI/demo/seg_and_pose_detect
bash -x build.sh
```	
### Setting Up the Target
Set up the target environment according to [Setting-up-the-target](../examples/Vitis-AI-Library#setting-up-the-target)
	 	  
### Running the demo

1. Copy the demo folder to the target using scp.
```
[Host]$scp -r demo root@IP_OF_BOARD:~/
```
2. Download the [vitis_ai_library_r1.3.x_video.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_library_r1.3.0_video.tar.gz). Copy them from host to the target using scp with the following command.
```
[Host]$scp vitis_ai_library_r1.3.x_video.tar.gz root@IP_OF_BOARD:~/
```
3. Untar the video packages on the target.
```
cd ~
tar -xzvf vitis_ai_library_r1.3.x_video.tar.gz 
```
4. Enter the directory of demo in target board and run it, take `seg_and_pose_detect` as an example.
```
cd ~/demo/seg_and_pose_detect
./seg_and_pose_detect_x seg_960_540.avi pose_960_540.avi -t 4 -t 4 >/dev/null 2>&1
```  
If you want to use DRM display, please connect to the board using SSH, and run the following command.
```
./seg_and_pose_detect_drm seg_960_540.avi pose_960_540.avi -t 4 -t 4 >/dev/null 2>&1
```
Note that, for demo with video input, only `webm` and `raw` format are supported by default with the official system image. 
If you want to support video data in other formats, you need to install the relevant packages on the system. 

5. For the demo with image input, such as `classification` and `yolov3` demo, please prepare the test images and execute the following command to run the demo.
```
#for classification demo
./demo_classification /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel resnet50_0 demo_classification.jpg

#for yolov3 demo
./demo_yolov3 /usr/share/vitis_ai_library/models/yolov3_voc/yolov3_voc.xmodel demo_yolov3.jpg
```

## Quick Start For Alveo
### Setting Up the Host for U50/U50lv/U280
1. Set up the host environment according to [Setting-up-the-host](../examples/Vitis-AI-Library#setting-up-the-host-for-u50u50lvu280)

2. To compile the demo, take `yolov3` as an example.
```
cd /workspace/demo/yolov3
bash -x build.sh
```	
### Running demo for U50/U50lv/U280
Suppose you have downloaded `Vitis-AI`, entered `Vitis-AI` directory, and then started Docker. 
Thus, demos are located in the path of `/workspace/demo/` in the docker system. 

**`/workspace/demo` is the path for the following example.**
 
1. Download the [vitis_ai_library_r1.3.0_video.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_library_r1.3.0_video.tar.gz) packages and untar it.
```
cd /workspace
wget https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_library_r1.3.0_video.tar.gz -O vitis_ai_library_r1.3.0_video.tar.gz
tar -xzvf vitis_ai_library_r1.3.0_video.tar.gz
```
2. Enter the directory of demo and then compile it. Take `seg_and_pose_detect` as an example.
```
cd /workspace/demo/seg_and_pose_detect
bash -x build.sh
```
3. Run the demo, take `seg_and_pose_detect` as an example.
```
cd demo/seg_and_pose_detect
./seg_and_pose_detect_x seg_960_540.avi pose_960_540.avi -t 4 -t 4 >/dev/null 2>&1
```
Note that DRM display mode is not supported for the cloud.  

4. For the demo with image input, such as `classification` and `yolov3` demo, please prepare the test images and execute the following command to run the demo.
```
#for classification demo
./demo_classification /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel resnet50_0 demo_classification.jpg

#for yolov3 demo
./demo_yolov3 /usr/share/vitis_ai_library/models/yolov3_voc/yolov3_voc.xmodel demo_yolov3.jpg
```

## Reference
For more information, please refer to [vitis-ai-library-user-guide](https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_3/ug1354-xilinx-ai-sdk.pdf).
