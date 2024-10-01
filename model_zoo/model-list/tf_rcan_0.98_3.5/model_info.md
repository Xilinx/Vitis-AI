# RCAN


### Contents
1. [Use Case and Application](#Use-Case-and-Application)
2. [Specification](#Specification)
3. [Paper and Architecture](#Paper-and-Architecture)
4. [Dataset Preparation](#Dataset-Preparation)
5. [Use Guide](#Use-Guide)
6. [License](#License)
7. [Note](#Note)


### Use Case and Application

   - Image Super Resolution
   - Trained on DIV2K dataset 
   
### Specification

| Metric             | Value                                   |
| :----------------- | :-------------------------------------- |
| Framework          | TensorFlow                              |
| Prune Ratio        | 98%                                     |
| FLOPs              | 86.95G                                  |
| Input Dims         | 360,640,3                               |
| FP32 Accuracy      | (Set5) 37.6402/0.9592                   |
| INT8 Accuracy      | (Set5) 37.2495/0.9556                   |
| Train Dataset      | DIV2K                                   |
| Test Dataset       | DIV2K                                   |
| Supported Platform | GPU, VEK280, V70                        |
  

### Paper and Architecture 

1. Network Architecture: RCAN

2. Paper link: https://arxiv.org/abs/1807.02758
  
  
### Dataset Preparation

1. Datasets description
   - [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
   - [Benchmarks](https://cv.snu.ac.kr/research/EDSR/benchmark.tar)
 
2. Dataset diretory structure
   + data
       + DIV2K
       + benchmark


### Use Guide

1. Model description
   light-wieght RCAN for scale=2x. Compared with the original version, the changes include:
   - Remove all Channel-attention (CA) moduel
   - Using less Residual blocks: --n_resgroups 3 --n_resblocks 2 --n_feats 32 

2. Evaluation
   ```
   # perform evaluation on 4 benchmarks: Set5, Set14, B100, Urabn100
   sh run_eval.sh
   ```
3. Training
   ```
   # perform training on DIV2K dataset
   sh run_train.sh
   ```
4. Testing on your own dataset
   ```
   # set the test dataset path as your own directory
   sh run_demo.sh
   ```

### License

Apache License 2.0

For details, please refer to **[Vitis-AI License](https://github.com/Xilinx/Vitis-AI/blob/master/LICENSE)**


### Note

Data preprocess

  ```
  data channel order: BGR(0~255)                  
  input = input / 255.0
  ```
  