# OFA depthwise resnet50


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
   - Searched by OFA (once-for-all) and support depthwise
   
   
### Specification

| Metric             | Value                                   |
| :----------------- | :-------------------------------------- |
| Framework          | PyTorch                                 |
| Prune Ratio        | 0%                                      |
| FLOPs              | 1.29G                                   |
| Input Dims (H W C) | 176,176,3                               |
| FP32 Accuracy      | 0.7633/0.9292 (top1/top5)               |
| INT8 Accuracy      | 0.7629/0.9306 (top1/top5)               |
| Train Dataset      | ImageNet ILSVRC2012 Train               |
| Test Dataset       | ImageNet ILSVRC2012 Val                 |
| Supported Platform | GPU, VEK280, V70                        |
  

### Paper and Architecture 

1. Network Architecture: ResNet50
 
2. Paper link: https://arxiv.org/abs/1512.03385
  
  
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

1. Evaluate float model. 
    ```shell
    cd code
    bash run_eval.sh
    ```

2. QAT(Quantization-Aware-Training)
    ```shell
    cd code
    bash run_qat.sh
    ```
    
    After QAT, use the following script to convert the QAT model, test the accuracy and dump xmodel for deployment.
    
    ```shell
    cd code
    bash run_qat_eval.sh
    ```

### License

Non-Commercial Use Only

For details, please refer to **[AMD license agreement for non commercial models](https://github.com/Xilinx/Vitis-AI/blob/master/model_zoo/AMD-license-agreement-for-non-commercial-models.md)**


### Note

data preprocess
  ```
  data channel order:RGB
  image_size_init: 320 x 320
  Resize 1: math.ceil(image_size_init / 0.875))
  crop: crop(iamge_size_init)
  Normalized_image = (image-mean)/std
    mean(0.485,0.456,0.406)
    std(0.229,0.224,0.225)
  Resize 2: image_size_bicubic=176x176
  ```
