# ResNet50


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
   - Searched by OFA (once-for-all) 
   
### Specification

| Metric             | Value                                   |
| :----------------- | :-------------------------------------- |
| Framework          | PyTorch                                 |
| Prune Ratio        | 88%                                     |
| FLOPs              | 1.8G                                    |
| Input Dims (H W C) | 160,160,3                               |
| FP32 Accuracy      | 0.758/0.926 (top1/top5)                 |
| INT8 Accuracy      | 0.748/0.921 (top1/top5)                 |
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

1. Evalaute float model.
    ```shell
    cd pt_ofa-resnet50_imagenet_192_192_0.74_3.6G_3.0/code
    bash run_eval.sh
    ```
2. Quantize model and evaluate.
    ```shell
    cd pt_ofa-resnet50_imagenet_192_192_0.74_3.6G_3.0/code
    bash run_quant.sh
    ```
3. Evalaute quantized model.
    ```shell
    cd pt_ofa-resnet50_imagenet_192_192_0.74_3.6G_3.0/code
    bash run_quant_eval.sh
    ```
	
### License

Non-Commercial Use Only

For details, please refer to **[AMD license agreement for non commercial models](https://github.com/Xilinx/Vitis-AI/blob/master/model_zoo/AMD-license-agreement-for-non-commercial-models.md)**


### Note

data preprocess
  ```
  data channel order:RGB
  image_size: 160 x 160
  Resize: math.ceil(image_size / 0.875))
  crop: crop(iamges_size)
  Normalized_image = (image-mean)/std
    mean(0.485,0.456,0.406)
    std(0.229,0.224,0.225)
  ```
