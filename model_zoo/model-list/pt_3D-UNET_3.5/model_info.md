# 3D UNet


### Contents
1. [Use Case and Application](#Use-Case-and-Application)
2. [Specification](#Specification)
3. [Paper and Architecture](#Paper-and-Architecture)
4. [Dataset Preparation](#Dataset-Preparation)
5. [Use Guide](#Use-Guide)
6. [License](#License)
7. [Note](#Note)


### Use Case and Application

   - 3D medical image segmentation task
   - Trained on 2019 Kidney Tumor Segmentation Challenge dataset called KiTS19
   
   
### Specification

| Metric             | Value                                   |
| :----------------- | :-------------------------------------- |
| Framework          | PyTorch                                 |
| Prune Ratio        | 0%                                      |
| FLOPs              | 1065.44G                                |
| Input Dims         | 128,128,128,1                           |
| FP32 Accuracy      | 0.8824 (Mean_Dice)                      |
| INT8 Accuracy      | 0.8774 (Mean_Dice)                      |
| Train Dataset      | KiTS19                                  |
| Test Dataset       | KiTS19                                  |
| Supported Platform | GPU                                     |
  

### Paper and Architecture 

1. Network Architecture: 3D UNet
 
2. Paper link: https://arxiv.org/abs/1606.06650
  
  
### Dataset Preparation

1. Download the data
   
    To download the data please follow the instructions:
    ```bash
    mkdir raw-data-dir
    cd raw-data-dir
    git clone https://github.com/neheller/kits19
    cd kits19
    pip3 install -r requirements.txt
    python3 -m starter_code.get_imaging
    ```
    This will download the original, non-interpolated data to `raw-data-dir/kits19/data`

 
2. Move to `raw-data-dir/kits19/data` to fixed position `data/raw_data` to run preprocessing/training/inference.    
    ```bash
    mkdir data
    mv raw-data-dir/kits19/data/* data/raw_data
    ```

3. Dataset direcotry structure 
    ```
    + data
        + raw_data
            + case_00000
            + case_00001
            + ...
    ```


### Use Guide

1. train from scratch
    ```bash
    bash run_train.sh
    ```

2. test fp32 model using pth
    ```bash
    bash run_test.sh
    ```

3. quantize and test the int8 model
    ```bash
    bash run_quant.sh
    ```


### License

Apache License 2.0

For details, please refer to **[Vitis-AI License](https://github.com/Xilinx/Vitis-AI/blob/master/LICENSE)**


### Note

The quality metric in this benchmark is mean (composite) DICE score for classes 1 (kidney) and 2 (kidney tumor). 
The metric is reported as `mean_dice` in the code.
