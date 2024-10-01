# face mask detection


### Contents
1. [Use Case and Application](#Use-Case-and-Application)
2. [Specification](#Specification)
3. [Paper and Architecture](#Paper-and-Architecture)
4. [Dataset Preparation](#Dataset-Preparation)
5. [Use Guide](#Use-Guide)
6. [License](#License)
7. [Note](#Note)


### Use Case and Application

   - Real-time video streaming mask detection
   - Trained on [mask-detector dataset](https://github.com/waittim/mask-detector/tree/master/modeling/data)
   
   
### Specification

| Metric             | Value                                   |
| :----------------- | :-------------------------------------- |
| Framework          | PyTorch                                 |
| Prune Ratio        | 0%                                      |
| FLOPs              | 0.67G                                   |
| Input Dims (H W C) | 512,512,3                               |
| FP32 Accuracy      | 0.886 MAP@0.5                           |
| INT8 Accuracy      | 0.881 MAP@0.5                           |   
| Train Dataset      | mask-detector dataset                   |
| Test Dataset       | mask-detector dataset                   |
| Supported Platform | GPU, VEK280, V70                        |
  

### Paper and Architecture 

1. Network Architecture: yolo-fastest
 
2. Paper link: https://arxiv.org/pdf/2101.00784.pdf
  
  
### Dataset Preparation

1. Dataset description
    - Download [face mask detection dataset](https://github.com/waittim/mask-detector/tree/master/modeling/data), then put it into `data`.

2. Dataset diretory structure
   ```
    + data
        + fmd
            + images
            + labels
            + face_mask.data
            + face_mask.names
            + train.shapes
            + train.txt
            + valid.shapes
            + valid.txt
            + labels.npy
        + samples
    ```


### Use Guide

1. Demo
    ```shell
    bash run_demo.sh
    ```
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
    bash run_qat.sh
    ```

### License

Apache License 2.0

For details, please refer to **[Vitis-AI License](https://github.com/Xilinx/Vitis-AI/blob/master/LICENSE)**


### Note

Data preprocess
  ```
  data channel order: BGR(0~255)                  
  resize: h * w = 512 * 512
  input = input / 255.0
  ``` 
