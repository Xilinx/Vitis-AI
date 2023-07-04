# RefineDet


### Contents
1. [Use Case and Application](#Use-Case-and-Application)
2. [Specification](#Specification)
3. [Paper and Architecture](#Paper-and-Architecture)
4. [Dataset Preparation](#Dataset-Preparation)
5. [Use Guide](#Use-Guide)
6. [License](#License)
7. [Note](#Note)


### Use Case and Application

   - RefineDet based on pruned Vgg16 for Medical Detection
   - Trained on EDD dataset 
   
### Specification

| Metric             | Value                                   |
| :----------------- | :-------------------------------------- |
| Framework          | TensorFlow                              |
| Prune Ratio        | 88%                                     |
| FLOPs              | 9.83G                                   |
| Input Dims         | 320,320,3                               |
| FP32 Accuracy      | 0.7839 mAP                              |
| INT8 Accuracy      | 0.8022 mAP                              |
| Train Dataset      | EDD                                     |
| Test Dataset       | EDD                                     |
| Supported Platform | GPU, VEK280, V70                        |
  

### Paper and Architecture 

1. Network Architecture: RefineDet

2. Paper link: https://arxiv.org/abs/1711.06897
  
  
### Dataset Preparation

- Download EDD2020 dataset from https://ieee-dataport.org/competitions/endoscopy-disease-detection-and-segmentation-edd2020#files
   - Extract the zip file with following command
     ```shell
     unzip EndoCV2020-Endoscopy-Disease-Detection-Segmentation-subChallenge_data.zip
     ```
   - Copy the "originalImage" folder to "./data/EDD/" folder as "images"
   - We select part images as the validtion set (refer the 'data/EDD/val.txt'), and collect related annotation to a txt file ('data/EDD/all_gt.txt')
   - The "./data" structure is as following:
   ```
    + data
        + EDD
          + images
          + all_gt.txt
          + val.txt
   ```


### Use Guide

```shell
sh run_eval.sh
```


### License

Apache License 2.0

For details, please refer to **[Vitis-AI License](https://github.com/Xilinx/Vitis-AI/blob/master/LICENSE)**


### Note

* input image size: 320\*320\*3 
* data channel order: RGB(0~255)    
* name of input node: 'image:0'
* name of output node: 'arm_cls:0', 'arm_loc:0', 'odm_cls:0', 'odm_loc:0'.


### Quantize

1. Quantize tool installation

   Please refer to [vai_q_tensorflow](../../../src/vai_quantizer/vai_q_tensorflow1.x)
  
2. Quantize workspace

   You could use code/quantize/ folder.
   
	
