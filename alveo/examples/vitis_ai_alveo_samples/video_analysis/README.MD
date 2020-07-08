This is ssd based video analysis example and it uses unified vitis-ai APIs to run on alveo platforms

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
  ```sh
  make
  ```

### Run
  ```sh
  ./run.sh <video path>
  ```

Input: mp4 video path

Output: Detection results saved as jpg images inside output folder in current path
> **Note:** Application assumes `vitis-ai-caffe` conda is already activated  

