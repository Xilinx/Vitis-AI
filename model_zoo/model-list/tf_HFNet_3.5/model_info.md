# HFNet


### Contents
1. [Use Case and Application](#Use-Case-and-Application)
2. [Specification](#Specification)
3. [Paper and Architecture](#Paper-and-Architecture)
4. [Dataset Preparation](#Dataset-Preparation)
5. [Use Guide](#Use-Guide)
6. [License](#License)
7. [Note](#Note)


### Use Case and Application

   - Robust Hierarchical Localization at Large Scale
   - Trained on Google Landmarks dataset 
   
   
### Specification

| Metric             | Value                                   |
| :----------------- | :-------------------------------------- |
| Framework          | TensorFlow                              |
| Prune Ratio        | 0%                                      |
| FLOPs              | 20.09G                                  |
| Input Dims (H W C) | 960,960,3                               |
| FP32 Accuracy      | 0.532/0.089/0.283                       |
| INT8 Accuracy      | 0.586/0.186/0.320 (QAT)                 |
| Train Dataset      | Google Landmarks                        |
| Test Dataset       | Google Landmarks                        |
| Supported Platform | GPU, VEK280                             |
  

### Paper and Architecture 

Network Architecture: HFNet

Paper link: https://arxiv.org/abs/1812.03506

  
### Dataset Preparation

Please refer to hfnet/settings.py to set data folder `$DATA_PATH` containing dataset and expriment folder `$EXPER_PATH` containing the trained models, training and evaluation logs and CNN predictions.
All datasets should be downloaded in `$DATA_PATH`. We give below additional details as well as the expected directory structures.


### Training

HF-Net is trained on the Google Landmarks datasets. First download the [index of images](https://github.com/ethz-asl/hierarchical_loc/releases/download/1.0/google_landmarks_index.csv) and then the dataset itself using the script `setup/scripts/download_google_landmarks.py`. You can stop the downloading when 185k images are downloaded.

```
google_landmarks/
└── images/
```

### Use Guide

1. Model description

   Standard HF-Net use official implementation. 

2. Training
   ```
   # perform training on Googlelandmarks dataset
   sh run_train.sh
   cd code
   sh export_pb.sh
   cd ..
   ```
3. Evaluation
   ```
   sh run_eval.sh
   ```

   
### License

Apache License 2.0

For details, please refer to **[Vitis-AI License](https://github.com/Xilinx/Vitis-AI/blob/master/LICENSE)**


### Note

1. Data preprocess

  ```
  data channel: Gray (0~255)                  
  input = (input - 128) / 128.0
  ```
  
2. Quantize tool installation

   Please refer to [vai_q_tensorflow](../../../src/vai_quantizer/vai_q_tensorflow1.x)
  
3. Quantize workspace

   You could use code/quantize/ folder.
   
   
   
