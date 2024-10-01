# Inceptionv3


### Contents
1. [Use Case and Application](#Use-Case-and-Application)
2. [Specification](#Specification)
3. [Paper and Architecture](#Paper-and-Architecture)
4. [Dataset Preparation](#Dataset-Preparation)
5. [Use Guide](#Use-Guide)
6. [License](#License)
7. [Note](#Note)


### Use Case and Application

   - Classic Image Classification
   - Trained on ImageNet dataset
   
   
### Specification

| Metric             | Value                                   |
| :----------------- | :-------------------------------------- |
| Framework          | PyTorch                                 |
| Prune Ratio        | 40%                                     |
| FLOPs              | 6.8G                                    |
| Param              | 16.35M                                   |
| Input Dims (H W C) | 299,299,3                               |
| FP32 Accuracy      | 0.768/0.931 (top1/top5)                 |
| INT8 Accuracy      | 0.764/0.929 (top1/top5)                 |
| Train Dataset      | ImageNet ILSVRC2012 Train               |
| Test Dataset       | ImageNet ILSVRC2012 Val                 |
| Supported Platform | GPU, VEK280, V70                        |
  

### Paper and Architecture 

1. Network Architecture: Inception-v3
 
2. Paper link: https://arxiv.org/abs/1512.00567
  
  
### Dataset Preparation

ImageNet dataset official link: [ImageNet](http://image-net.org/download-images)

  ```
  The downloaded ImageNet dataset needs to be organized as the following format:
    1) Create a folder of "data". Put the validation data into folder: 'data/'.
    2) Data from each class for validation set needs to be put in the folder:
    +data/Imagenet/val
         +val/n01847000
         +val/n02277742
         +val/n02808304
         +...
  ```


### Use Guide

1. Evaluate float model
  ```shell
  cd code
  sh run_test_float.sh
  ```
2. Evaluate quantized(INT8) model
  ```shell
  sh run_test_quantized.sh
  ```

### License

Non-Commercial Use Only

For details, please refer to **[AMD license agreement for non commercial models](https://github.com/Xilinx/Vitis-AI/blob/master/model_zoo/AMD-license-agreement-for-non-commercial-models.md)**


### Note

Data preprocess
  ```
  data channel order: BGR(0~255)
  resize: short side reisze to 256 and keep the aspect ratio
  center crop: 299 * 299
  ```
