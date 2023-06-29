# PointPillars


### Contents
1. [Use Case and Application](#Use-Case-and-Application)
2. [Specification](#Specification)
3. [Paper and Architecture](#Paper-and-Architecture)
4. [Dataset Preparation](#Dataset-Preparation)
5. [Use Guide](#Use-Guide)
6. [License](#License)
7. [Note](#Note)


### Use Case and Application

   - 3D Object Detection
   - Trained on KITTI dataset
   
   
### Specification

| Metric             | Value                                   |
| :----------------- | :-------------------------------------- |
| Framework          | PyTorch                                 |
| Prune Ratio        | 0%                                      |
| FLOPs              | 11.2G                                   |
| Input Dims         | 12000,100,4                             |
| FP32 Accuracy      | Car 3D AP@0.5:90.79, 89.66, 88.78       |
| INT8 Accuracy      | Car 3D AP@0.5:90.80, 89.74, 88.80       |
| Train Dataset      | KITTI                                   |
| Test Dataset       | KITTI                                   |
| Supported Platform | GPU, VEK280, V70                        |
  

### Paper and Architecture 

1. Network Architecture: PointPillars
 
2. Paper link: https://arxiv.org/abs/1812.05784

    
### Dataset Preparation

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


### Use Guide

1. Evaluation
  - Configure the config.proto file.
  ```shell
  ...
  eval_input_reader: {
    ...
    kitti_info_path: "data/KITTI/kitti_infos_val.pkl"
    kitti_root_path: "data/KITTI"
  }
  ```
  - Execute run_eval.sh.
  ```shell
  bash code/test/run_eval.sh
  ```

2. Training
  - Configure the config.proto file.
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
  - Execute run_train.sh.
  ```shell
  bash code/train/run_train.sh
  ```

3. Model quantization
  ```shell
  bash code/test/run_quant.sh
  ```

4. QAT(Quantization-Aware-Training)
  - Configure the variables in `code/qat/run_qat.sh` and run the script.
  ```shell
  bash code/qat/run_qat.sh
  ```
  - After QAT, use the following script to convert the QAT model, test the accuracy and dump xmodel for deployment.
  ```shell
  bash code/qat/convert_test_qat.sh
  ```

### License

Apache License 2.0

For details, please refer to **[Vitis-AI License](https://github.com/Xilinx/Vitis-AI/blob/master/LICENSE)**

### Model_info

1. Data preprocess
  ```
  Voxelization on BEV -> pillars
  Utilize PointNet on each pillars
  Generate peseudo BEV image
  ``` 

### Acknowledgement
This repo comes from [PointPillars](https://github.com/nutonomy/second.pytorch.git), many thanks to them for their contribution to the community.

  