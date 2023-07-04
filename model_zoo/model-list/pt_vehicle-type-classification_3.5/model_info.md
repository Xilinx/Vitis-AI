# Vehicle Type Classification


### Contents
1. [Use Case and Application](#Use-Case-and-Application)
2. [Specification](#Specification)
3. [Paper and Architecture](#Paper-and-Architecture)
4. [Dataset Preparation](#Dataset-Preparation)
5. [Use Guide](#Use-Guide)
6. [License](#License)
7. [Note](#Note)


### Use Case and Application

   - Vehicle Type Classification
   - Trained on CarBodyStyle dataset
   
   
### Specification

| Metric             | Value                                   |
| :----------------- | :-------------------------------------- |
| Framework          | PyTorch                                 |
| Prune Ratio        | 0%                                      |
| FLOPs              | 3.64G                                   |
| Input Dims (H W C) | 224,224,3                               |
| FP32 Accuracy      | 0.8482                                  |
| INT8 Accuracy      | 0.8467                                  |
| Train Dataset      | CarBodyStyle                            |
| Test Dataset       | CarBodyStyle                            |
| Supported Platform | GPU, VEK280, V70                        |
  
 
### Dataset Preparation

1. Dataset description
    - Download [CarBodyStyle](https://www.kaggle.com/datasets/darshan1504/car-body-style-dataset) dataset, then unzip and put it into `data`.
    - Use object detection model crop car box from image to build `cropped_CarBodyStyle`.

2. Dataset diretory structure
   ```
    + data
        + cropped_CarBodyStyle
    ```
3. Remove broken images and split dataset to trainset and testset
    ```
    python code/split_dataset.py
    ```
4. Dataset diretory structure
   ```
    + data
        + cropped_CarBodyStyple
        + cropped_CarBodyStyple_splited
            + train
                + Buggy, Convertible, Coupe, Hatchback, Limousine, Minivan, Sedan
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