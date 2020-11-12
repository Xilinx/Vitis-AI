### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Performance](#performance)
5. [Model_info](#model_info)
6. [Acknowledgement](#acknowledgement)

### Installation

1. Environment requirement
    - anaconda3
    - python 3.6
    - pytorch, torchvision, tensorboardX, shapely, pybind11, protobuf, scikit-image, numba, pillow, google-sparsehash, fire etc.
    - vai_q_pytorch(Optional, required for quantization)
    - XIR Python frontend (Optional, required for dumping xmodel)

2. Installation with Docker

   First refer to [vitis-ai](https://github.com/Xilinx/Vitis-AI/tree/master/) to obtain the docker image.
   ```bash
   conda create -n pointpillars-env --clone vitis-ai-pytorch
   conda activate pointpillars-env
   conda install google-sparsehash -c bioconda
   pip install -r requirements.txt

   # Install Boost geometry and cuda-toolkit
   sudo apt-get update && sudo apt-get install libboost-all-dev && sudo apt-get install nvidia-cuda-toolkit

   # Export the following environment variables for numba
   export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
   export NUMBAPRO_NVVM=/usr/lib/x86_64-linux-gnu/libnvvm.so
   export NUMBAPRO_LIBDEVICE=/usr/lib/cuda/nvvm/libdevice
   export CUDA_HOME=/usr/lib/cuda

   # for testing
   export PYTHONPATH=/YourPath/pointpillars/code/test:$PYTHONPATH
   # for training
   export PYTHONPATH=/YourPath/pointpillars/code/train:$PYTHONPATH
   ```

### Preparation

1. Dataset description

Based on the KITTI dataset, 3 classes are used: Car, Pedestrian and Cyclist. The original KITTI training set includes 7481 samples, which is split into two parts: training set = 3712, validation set = 3769.

2. Download KITTI dataset and create some directories first:
  ```plain
  └── data
       └── KITTI
             ├── training    <-- 7481 train data
             |   ├── image_2 <-- for visualization
             |   ├── calib
             |   ├── label_2
             |   ├── velodyne
             |   └── velodyne_reduced <-- empty directory
             └── testing     <-- 7580 test data
                 ├── image_2 <-- for visualization
                 ├── calib
                 ├── velodyne
                 └── velodyne_reduced <-- empty directory
  ```
  Create kitti infos for validation and training:
  ```bash
  bash code/test/prepare_data.sh
  ```

### Train/Eval

1. Evaluation
  Configure the config.proto file.
  ```shell
  ...
  eval_input_reader: {
    ...
    kitti_info_path: "data/KITTI/kitti_infos_val.pkl"
    kitti_root_path: "data/KITTI"
  }
  ```
  Modify the PATH in run_eval.sh and run the script.
  ```shell
  bash code/test/run_eval.sh
  ```

2. Training
  Configure the config.proto file.
  ```shell
  ...
  train_input_reader: {
    database_sampler {
      database_info_path: "data/KITTI/kitti_dbinfos_train.pkl"
      ...
    }
    ...
    kitti_info_path: "data/KITTI/kitti_infos_train.pkl"
    kitti_root_path: "data/KITTI"
  }
  ```
  Modify the PATH and assign working directory in run_train.sh and run the script.
  ```shell
  bash code/train/run_train.sh
  ```

3. Model quantization
  ```shell
  bash code/test/run_quant.sh
  ```

### Performance
|Metric | Float | Quantized |
| -     | -    | - |
|Car BEV AP@0.7(easy, moderate, hard)|90.02, 81.26, 80.34|89.52, 79.46, 76.58|
|Car 3D AP@0.7(easy, moderate, hard)|81.69, 69.25, 66.46|76.01, 64.63, 57.70|
|Car BEV AP@0.5(easy, moderate, hard)|90.80, 89.92, 89.22|90.78, 88.03, 85.34|
|Car 3D AP@0.5(easy, moderate, hard)|90.79, 89.66, 88.78|90.75, 87.04, 83.44|
|Pedestrian BEV AP@0.5(easy, moderate, hard)|53.63, 48.51, 45.29|50.95, 46.13, 42.53|
|Pedestrian 3D AP@0.5(easy, moderate, hard)|46.48, 41.56, 38.70|42.67, 38.12, 34.57|
|Cyclist BEV AP@0.5(easy, moderate, hard)|71.65, 53.61, 50.36|65.26, 49.71, 46.97|
|Cyclist 3D AP@0.5(easy, moderate, hard)|69.17, 51.03, 48.02|59.22, 44.97, 42.48|


### Model_info

1. Data preprocess
  ```
  Voxelization on BEV -> pillars
  Utilize PointNet on each pillars
  Generate peseudo BEV image
  ``` 

### Acknowledgement
This repo comes from [PointPillars](https://github.com/nutonomy/second.pytorch.git), many thanks to them for their contribution to the community.
