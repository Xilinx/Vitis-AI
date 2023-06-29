# VisionTransformer


### Contents
1. [Use Case and Application](#Use-Case-and-Application)
2. [Specification](#Specification)
3. [Paper and Architecture](#Paper-and-Architecture)
4. [Dataset Preparation](#Dataset-Preparation)
5. [Use Guide](#Use-Guide)
6. [License](#License)
7. [Note](#Note)


### Use Case and Application

   - Image Classification
   - Trained on ImageNet dataset
   
   
### Specification

| Metric             | Value                                   |
| :----------------- | :-------------------------------------- |
| Framework          | TensorFlow                              |
| Prune Ratio        | 0%                                      |
| FLOPs              | 21.3G                                   |
| Input Dims (H W C) | 352,352,3                               |
| FP32 Accuracy      | 0.8282                                  |
| INT8 Accuracy      | 0.8254                                  |
| Train Dataset      | ImageNet ILSVRC2012 Train               |
| Test Dataset       | ImageNet ILSVRC2012 Val                 |
| Supported Platform | GPU                                     |
  

### Paper and Architecture 

1. Network Architecture: ViT
 
2. Paper link: https://arxiv.org/abs/2010.11929
  
  
### Dataset Preparation

1. Dataset description
    - Download [imagenet 2012](https://image-net.org/download.php) dataset, then unzip and put it into `data`.

2. Dataset diretory structure
   ```
    + data
        + test
    ```

### Use Guide

1. Float Evaluation
    ```shell
    bash run_test.sh
    ```
2. Quant Evaluation 
    ```shell
    bash run_quant.sh
    ```
	
### License

Non-Commercial Use Only

For details, please refer to **[AMD license agreement for non commercial models](https://github.com/Xilinx/Vitis-AI/blob/master/model_zoo/AMD-license-agreement-for-non-commercial-models.md)**


### Note

1. Data preprocess
  
  ```
  data channel order: BGR(0~255)                  
  pad_and_resize: h * w = 352 * 352
  input = (input - 127.5) / 127.5
  ```

2. ViT-352 finetuned from [ViT official B32_pretrained_model](https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz)
+ patch_size=32
+ input_size=352
+ sequence_len=11*11+1=122

3. Deployment Demo

https://github.com/Xilinx/Vitis-AI/tree/2.5/examples/Transformer
  
  



  
  
