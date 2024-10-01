# EfficientNet-EdgeTPU-L


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
| Framework          | TensorFlow                              |
| Prune Ratio        | 0%                                      |
| FLOPs              | 19.36G                                  |
| Input Dims (H W C) | 300,300,3                               |
| FP32 Accuracy      | 0.8026/0.9514 (top1/top5)               |
| INT8 Accuracy      | 0.7996/0.9491 (top1/top5)               |
| Train Dataset      | ImageNet ILSVRC2012 Train               |
| Test Dataset       | ImageNet ILSVRC2012 Val                 |
| Supported Platform | GPU, VEK280, V70                        |
  

### Paper and Architecture 

1. Network Architecture: EfficientNet-EdgeTPU Large
 
2. Paper link: https://arxiv.org/abs/2003.02838
  
  
### Dataset Preparation

1. dataset describe.
  ```
  dataset includes image file and groundtruth file.
  image file: put train/val images.
  groundtruth file: format as "imagename label"
  ```
2. prepare testdatset.
   
  ```
  cd code/gen_data

  # check soft link vaild or 
  # user download validation dataset, rename it with "validation".

  # dataset put or generated in data fileholder.
  # validation dataset fortmat: A big file(validation) includes 1000 subfiles, echo subfile put same class images.  
  # run get_dataset.sh, then will generate required dataset for model.

  bash get_dataset.sh 
  ```


### Use Guide

1. Evaluate ckpt model.
  ```shell
  bash code/test/eval.sh
  ```
  
2. Evaluate float pb model.
  ```shell
  cd code/test/
  bash run_eval_pb.sh
  ```
  
### License

Non-Commercial Use Only

For details, please refer to **[AMD license agreement for non commercial models](https://github.com/Xilinx/Vitis-AI/blob/master/model_zoo/AMD-license-agreement-for-non-commercial-models.md)**


### Note

1. Data preprocess
  ```
  data channel order: RGB(0~255)                  
  crop: crop the central region of the image. 
  resize: 300 * 300 (tf.image.resize_bicubic) 
  ``` 
2. Node information

  ```
  The name of input node: 'input:0'
  The name of output node: 'logits:0'
  ```
3. Quantize tool installation

   Please refer to [vai_q_tensorflow](../../../src/vai_quantizer/vai_q_tensorflow1.x)
  
4. Quantize workspace

   You could use code/quantize/ folder.  
