# ResNet-50 TensorFlow example

This example runs resnet50_v1 tensorflow using unified Vitis-AI APIs to run on Alveo-U200/U250 platforms.

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
./run.sh
```

`Input`: It takes input as jpg image from `image` folder in current path.
`Output`: Results are displayed on the console.
