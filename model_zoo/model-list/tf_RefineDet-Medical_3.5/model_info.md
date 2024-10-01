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
| Prune Ratio        | 0%                                      |
| FLOPs              | 81.28G                                  |
| Input Dims         | 320,320,3                               |
| FP32 Accuracy      | 0.7866 mAP                              |
| INT8 Accuracy      | 0.7857 mAP                              |
| Train Dataset      | EDD                                     |
| Test Dataset       | EDD                                     |
| Supported Platform | GPU, VEK280, V70                        |
  

### Paper and Architecture 

1. Network Architecture: RefineDet

2. Paper link: https://arxiv.org/abs/1711.06897
  
  
### Dataset Preparation

1. Dataset pre-processing.
   Please make soft link of the EDD dataset

   ```
    + data
        + EDD
          + images
          + bbox
          + all_gt.txt
          + val.txt
   ```
   
   
### Use Guide

1. Evaluate the tf pb model
```shell
sh run_eval.sh
```

2. Visulization for the tf pb model.
```shell
cd code/test
python demo.py
```

### License

Apache License 2.0

For details, please refer to **[Vitis-AI License](https://github.com/Xilinx/Vitis-AI/blob/master/LICENSE)**


### Note

* input image size: 320\*320 
* data channel order: RGB(0~255)    
* name of input node: 'image:0'
* name of output node: 'arm_cls:0', 'arm_loc:0', 'odm_cls:0', 'odm_loc:0'.


### Quantize

1. Quantize tool installation

   Please refer to [vai_q_tensorflow](../../../src/vai_quantizer/vai_q_tensorflow1.x)
  
2. Quantize workspace

   You could use code/quantize/ folder.
   