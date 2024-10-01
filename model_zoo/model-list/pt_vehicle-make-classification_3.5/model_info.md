# Vehicle Make Classification


### Contents
1. [Use Case and Application](#Use-Case-and-Application)
2. [Specification](#Specification)
3. [Paper and Architecture](#Paper-and-Architecture)
4. [Dataset Preparation](#Dataset-Preparation)
5. [Use Guide](#Use-Guide)
6. [License](#License)
7. [Note](#Note)


### Use Case and Application

   - Vehicle Make Classification
   - Trained on VMMR dataset
   
   
### Specification

| Metric             | Value                                   |
| :----------------- | :-------------------------------------- |
| Framework          | PyTorch                                 |
| Prune Ratio        | 0%                                      |
| FLOPs              | 3.64G                                   |
| Input Dims (H W C) | 224,224,3                               |
| FP32 Accuracy      | 0.9536                                  |
| INT8 Accuracy      | 0.9522                                  |
| Train Dataset      | VMMR                                    |
| Test Dataset       | VMMR                                    |
| Supported Platform | GPU, VEK280, V70                        |
  
 
### Dataset Preparation

1. Dataset description
    - Download [VMMRdb](https://github.com/faezetta/VMMRdb) dataset, then unzip and put it into `data`.
    - Use object detection model crop car box from image to build `cropped_VMMRdb`.

2. Dataset diretory structure
   ```
    + data
        + cropped_VMMRdb
    ```
3. Reorg dataset and split dataset to trainset and testset
    ```
    python code/split_for_make.py
    ```
4. Dataset diretory structure
   ```
    + data
        + cropped_VMMRdb
        + cropped_VMMRdb_splited
            + train
                + acura, audi, bmw, buick, cadillac, chevrolet, chrysler, dodge, ford, gmc, honda, hummer, hyundai, infiniti, isuzu, jaguar, jeep, kia, landrover, lexus, lincoln, mazda, mercedes_benz, mercury, mini, mitsubishi, nissan, oldsmobile, plymouth, pontiac, porsche, saab, saturn, scion, subaru, suzuki, toyota, volkswagen, volvo
            + test
    ```
    


### Use Guide

1. Evaluation
    ```shell
    bash run_test.sh
    ```
2. Training
    ```shell
    bash run_train.sh
    ```
3. Quantization
    ```shell
    bash run_quant.sh
    ```



### License

Apache License 2.0

For details, please refer to **[Vitis-AI License](https://github.com/Xilinx/Vitis-AI/blob/master/LICENSE)**


### Note

Data preprocess
  ```
  data channel order: BGR(0~255)                  
  resize: h * w = 224 * 224
  input = input / 255
  ``` 