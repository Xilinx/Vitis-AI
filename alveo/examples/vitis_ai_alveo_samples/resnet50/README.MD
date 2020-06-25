This is resnet50_v1 tensorflow example and it uses unified vitis-ai APIs to run on alveo platforms

### Setup

> **Note:** Skip, If you have already run the below steps.

Activate Conda Environment
  ```sh
  conda activate vitis-ai-caffe
  ```

Setup the Environment

  ```sh
  source /workspace/alveo/overlaybins/setup.sh
  ```

### Build

    make

### Run

  ```sh
  source run.sh
  ```

Input: It takes input as jpg image from image folder in current path

Output: classification results display on console window

> **Note:** Application assumes `vitis-ai-caffe` conda is already activated  

