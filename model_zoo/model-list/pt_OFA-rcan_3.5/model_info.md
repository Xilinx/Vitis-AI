# OFA RCAN


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
   - Searched by OFA (once-for-all) 
   
### Specification

| Metric             | Value                                   |
| :----------------- | :-------------------------------------- |
| Framework          | PyTorch                                 |
| Prune Ratio        | 0%                                      |
| FLOPs              | 40.5G                                   |
| Input Dims         | 360,640,3                               |
| FP32 Accuracy      | (Set5) PSNR/SSIM= 37.654/0.959          |
| INT8 Accuracy      | (Set5) PSNR/SSIM= 37.384/0.956          |
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
   ofa-rcan based on ofa search method under latency constraint (vck190 latency constraint=59ms) for scale=2x. 

2. Evaluation
   ```
   # perform evaluation on 4 benchmarks: Set5, Set14, B100, Urban100
   cd code/
   sh run_test.sh
   ```
3. Training
   ```
   # perform training on DIV2K dataset. Training is based on a pretrained model which is generated using OFA under latency constraint.
   cd code/
   sh run_train.sh
   ```
4. Quantization
   ```
   # perform qat on DIV2K dataset
   cd code/
   sh run_qat.sh
   ```

### License

Apache License 2.0

For details, please refer to **[Vitis-AI License](https://github.com/Xilinx/Vitis-AI/blob/master/LICENSE)**


### Note

Data preprocess
  ```
  data channel order: BGR(0~255)                  
  input = input 
  ```
  