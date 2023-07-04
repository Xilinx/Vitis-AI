# 2D UNet

### Contents
1. [Use Case and Application](#Use-Case-and-Application)
2. [Specification](#Specification)
3. [Paper and Architecture](#Paper-and-Architecture)
4. [Dataset Preparation](#Dataset-Preparation)
5. [Use Guide](#Use-Guide)
6. [License](#License)
7. [Note](#Note)


### Use Case and Application

   - 2D medical image segmentation task
   - Trained on BraTS Dataset
   
   
### Specification

| Metric             | Value                                   |
| :----------------- | :-------------------------------------- |
| Framework          | TensorFlow2                             |
| Prune Ratio        | 0%                                      |
| FLOPs              | 24.6G                                   |
| Input Dims         | 144,144,4                               |
| FP32 Accuracy      | Dice_coef 0.8749                        |
| INT8 Accuracy      | Dice_coef 0.8735                        |
| Train Dataset      | BraTS                                   |
| Test Dataset       | BraTS                                   |
| Supported Platform | GPU, VEK280, V70                        |
  

### Paper and Architecture 

Network Architecture: 2D UNet 
  
  
### Dataset Preparation

1. Prepare datset.

  - Dataset link: [BraTS subset](https://drive.google.com/file/d/1A2IU8Sgea1h3fYLpYtFb2v7NYdMjvEhU/view) 
  ```shell  
  a. Users need to download the BraTS dataset by themselves.
  b. The downloaded BraTS dataset needs to be organized as the following format:
    1) Make a folder "decathlon" under "data" folder directory and put the download dataset here (e.g. `./data/decathlon/Task01_BrainTumour.tar`)
    2) Untar the "Task01_BrainTumour.tar" file (e.g. `tar -xvf Task01_BrainTumour.tar`)
  ```
2. Data pre-processing.
  - Run script to convert to a single HDF5 file 
  ```shell
  cd ./code/
  bash prepare_data.sh
  ```

  * `data` directory structure like:
     + decathlon
       + Task01_BrainTumour.h5
       + Task01_BrainTumour
         + dataset.json  
         + imagesTr
         + imagesTs
         + labelsTr



### Use Guide

1. Evaluate model.
  ```shell
  bash run_test.sh
  ```

2. Training model.
  ```shell
  bash run_train.sh
  ```
  

### Quantization
**vai_q_tensorflow2** is required, see
[Vitis AI User Document](https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html#documentation) for installation guide.

1. Quantize model.
  ```shell
  bash run_quantize.sh
  ```

2. Evaluate quantized model.
  ```shell
  bash run_quantize_test.sh
  ```

3. Dump quantized golden results.
  ```shell
  bash run_quantize_dump.sh
  ```


### License

Apache License 2.0

For details, please refer to **[Vitis-AI License](https://github.com/Xilinx/Vitis-AI/blob/master/LICENSE)**


### Note

1. Data preprocess

  ```
   data channel order: RGB(0~255) 
  ``` 
  
2. The quality metric in this benchmark is mean (composite) DICE score for classes 1 (kidney) and 2 (kidney tumor). 

   The metric is reported as `mean_dice` in the code.
