# Segmentation

This Segmentation example uses unified Vitis-AI APIs to run on Alveo-U200/U250 platforms.

## Setup

:pushpin: **Note:** Skip, If you have already run the below steps.

Activate Conda Environment

```sh
conda activate vitis-ai-caffe
```
Setup the Environment

```sh
source /vitis_ai_home/setup/alveo/u200_u250/overlaybins/setup.sh
```

## Build

```sh
make
```

## Run

```sh
./run.sh <video path>
```

:pushpin: **Note:** User needs to provide the video path or sample video can be downloaded from [here](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_runtime_r1.3.0_image_video.tar.gz).

The output detection results are saved as `.jpg` images inside output folder in current path.
