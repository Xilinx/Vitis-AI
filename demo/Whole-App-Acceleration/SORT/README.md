## SORT based Multi-Object Tracking
:pushpin: **Note:** This application can be run only on ZCU102 platform.

## Table of Contents

- [Introduction](#Introduction)
- [Setup the target platform](#Setup-the-target-platform)
- [Build and running the application](#Build-and-running-the-application)
- [Performance](#Performance)    

## Introduction
SORT is Simple Online & Realtime tracking , used for multi object tracking. It performs Kalman filtering in image space and frame by frame data association using Hungarian method with an association matrix that measures bouding box overlap. 

https://github.com/mcximing/sort-cpp has original code base, here all the subfunctions like Kalman filter & Hungarian method are running on CPU. In this tutorial, accelerated Kalman filter is provided for the SORT and can only be run  on ZCU102 board (device part xczu9eg-ffvb1156-2-e).

## Setup the target platform
* Installing a Board Image.
	* Download the SD card system image files from the following links:  
	
      [ZCU102](https://www.xilinx.com/member/forms/download/xef.html?filename=xilinx-zcu102-sort-v2020.2-v1.4.0.img.gz)   

      	Note: The version of the board image should be 2020.2 or above.
	* Use Etcher software to burn the image file onto the SD card.

## Build and running the application

### (Optional) Build the application
Commands to run -
    
    source < path-to-Vitis-installation-directory >/settings64.sh
    source < path-to-XRT-installation-directory >/setup.sh
    export DEVICE=< path-to-platform-directory >/xilinx_zcu102_base_202020_1.xpfm

Download the platform, and common-image from Xilinx Download Center. Run the sdk.sh script from the common-image directory to install sysroot using the command : "./sdk.sh -y -d ./ -p"

Unzip the rootfs file : "gunzip ./rootfs.ext4.gz"

    export SYSROOT=< path-to-platform-sysroot >
    export EDGE_COMMON_SW=< path-to-rootfs-and-Image-files >
    export PERL=<path-to-perl-installation-location> #For example, "export PERL=/usr/bin/perl". Please make sure that Expect.pm package is available in your Perl installation.
    cd < Vitis-AI-path >/demo/Whole-App-Acceleration/SORT
    make host TARGET=hw HOST_ARCH=aarch64

Above make instruction generates executable file at "build_dir.hw.xilinx_zcu102_base_202020_1/sort.exe". Copy this exe file to SD card.

### Data Preparation
- Download MOT16 dataset at https://motchallenge.net/data/MOT16.zip. Copy any dataset, for example test/MOT16-01 to SD card. 
> **Note:** User is responsible for the use of the downloaded content and compliance with any copyright licenses.

### Running the Application
This part is about how to run the SORT application on zcu102 board. Please insert SD_CARD on the ZCU102 board.After the linux boot, run:

```
% cd /media/sd-mmcblk0p1
% export XILINX_XRT=/usr
% echo 1 > /proc/sys/kernel/printk
```
* Run SORT without waa
```
% ./sort.exe ./MOT16-01 0

Expect: 
Processing ./MOT16-01...
Total Tracking took: 0.603764 for 450 frames or 745.324 FPS
```
* Run SORT with waa
```
% ./sort.exe ./MOT16-01 1

Expect: 
Processing ./MOT16-01...
Total Tracking took: 0.381255 for 450 frames or 1180.311 FPS
```

## Performance:
Below table shows the comparison of througput achieved for SORT by accelerating the Kalman filter on ZCU102 board.

<table style="undefined;table-layout: fixed; width: 534px">
<colgroup>
<col style="width: 136px">
<col style="width: 136px">
<col style="width: 145px">
<col style="width: 134px">
</colgroup>
  <tr>
    <th rowspan="2">Dataset</th>
    <th colspan="2">FPS</th>
    <th rowspan="2"><span style="font-weight:bold">Percentage improvement in throughput</span></th>
  </tr>
  <tr>
    <td>with software KalmanFilter</td>
    <td>with hardware accelerated KalmanFilter</td>
  </tr>

  <tr>
   <td>test/MOT16-01</td>
    <td>745.3</td>
    <td>1180.3</td>
        <td>58.36%</td>
  </tr>

  <tr>
   <td>test/MOT16-03</td>
    <td>119.6</td>
    <td>186.3</td>
        <td>55.76%</td>
  </tr>

   <tr>
   <td>test/MOT16-06</td>
    <td>615</td>
    <td>937.1</td>
        <td>52.37%</td>
  </tr>

   <tr>
   <td>test/MOT16-07</td>
    <td>212.7</td>
    <td>330.7</td>
        <td>55.47%</td>
  </tr>
  
   <tr>
   <td>test/MOT16-08</td>
    <td>427.7</td>
    <td>694.3</td>
        <td>62.33%</td>
  </tr>

   <tr>
   <td>test/MOT16-12</td>
    <td>502.8</td>
    <td>769.7</td>
        <td>53.08%</td>
  </tr>

   <tr>
   <td>test/MOT16-14</td>
    <td>274.7</td>
    <td>455.5</td>
        <td>65.81%</td>
  </tr>

   <tr>
   <td>train/MOT16-02</td>
    <td>601.3</td>
    <td>974.1</td>
        <td>61.99%</td>
  </tr>

   <tr>
   <td>train/MOT16-04</td>
    <td>201.1</td>
    <td>330.2</td>
        <td>64.19%</td>
  </tr>

   <tr>
   <td>train/MOT16-05</td>
    <td>899.3</td>
    <td>1339.5</td>
        <td>48.94%</td>
  </tr>

   <tr>
   <td>train/MOT16-09</td>
    <td>599.4</td>
    <td>960.4</td>
        <td>60.22%</td>
  </tr>

   <tr>
   <td>train/MOT16-10</td>
    <td>324.3</td>
    <td>505.8</td>
        <td>55.96%</td>
  </tr>
  
   <tr>
   <td>train/MOT16-11</td>
    <td>616.4</td>
    <td>963.8</td>
        <td>56.35%</td>
  </tr>
  
   <tr>
   <td>train/MOT16-13</td>
    <td>401.5</td>
    <td>754.7</td>
        <td>87.97%</td>
  </tr>
</table>
