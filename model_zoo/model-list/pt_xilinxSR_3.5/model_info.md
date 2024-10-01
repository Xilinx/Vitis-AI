# xilinxSR


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
| Framework          | PyTorch                                 |
| Prune Ratio        | 0%                                      |
| FLOPs              | 364.9G                                  |
| Input Dims         | 360,640,3                               |
| FP32 Accuracy      | 29.04dB                                 |
| INT8 Accuracy      | 28.66dB                                 |
| Train Dataset      | DIV2K                                   |
| Test Dataset       | DIV2K                                   |
| Supported Platform | GPU, VEK280                             |
  
 
### Dataset Preparation

1. Datasets description
   - [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
 
2. Dataset structure
   ```
   + data
       + DIV2K
   ```



### Use Guide

1. Model description
   xilinxSR acheved best PSNR under constrained flops and paramaters for scale=4x. 

2. Evaluation
   ```
   # perform evaluation on DIV2K validation set (801-900)
   sh run_test.sh
   ```
3. Post Quantization (PTQ)
   ```
   # perform PTQ on DIV2K dataset
   sh run_quant.sh
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
  