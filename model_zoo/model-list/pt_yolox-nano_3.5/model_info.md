# Modified YOLOX-Nano


### Contents
1. [Use Case and Application](#Use-Case-and-Application)
2. [Specification](#Specification)
3. [Paper and Architecture](#Paper-and-Architecture)
4. [Dataset Preparation](#Dataset-Preparation)
5. [Use Guide](#Use-Guide)
6. [License](#License)
7. [Note](#Note)


### Use Case and Application

   - Classic Object Detection
   - Trained on COCO dataset 
   
   
### Specification

| Metric             | Value                                   |
| :----------------- | :-------------------------------------- |
| Framework          | PyTorch                                 |
| Prune Ratio        | 0%                                      |
| FLOPs              | 1G                                      |
| Input Dims (H W C) | 416,416,3                               |
| FP32 Accuracy      | 0.22 mAP                                |
| INT8 Accuracy      | 0.21 mAP (QAT)                          |
| Train Dataset      | COCO2017                                |
| Test Dataset       | COCO2017                                |
| Supported Platform | GPU, VEK280, V70                        |
  

### Paper and Architecture 

Network Architecture: YOLOX-Nano

Paper link: https://arxiv.org/abs/2107.08430

  
### Dataset Preparation

1. Dataset description

The dataset MSCOCO2017 contains 118287 images for training and 5000 images for validation.

2. Download COCO dataset and create directories like this:
  ```plain
  └── data
       └── COCO
             ├── annotations
             |   ├── instances_train2017.json
             |   ├── instances_val2017.json
             |   └── ...
             ├── train2017
             |   ├── 000000000009.jpg
             |   ├── 000000000025.jpg
             |   ├── ...
             ├── val2017
                 ├── 000000000139.jpg
                 ├── 000000000285.jpg
             |   ├── ...
             └── test2017
                 ├── 000000000001.jpg
                 ├── 000000000016.jpg
                 └── ...
  ```


### Use Guide

1. Evaluation
  - Execute run_eval.sh.
  ```shell
  bash code/run_eval.sh
  ```

2. Training
  ```shell
  bash code/run_train.sh
  ```

3. Model quantization and xmodel dumping
  ```shell
  bash code/run_quant.sh
  ```

4. QAT(Quantization-Aware-Training), model converting and xmodel dumping
  - Configure the variables and in `code/run_qat.sh` and `code/exps/example/custom/yolox_nano_deploy_relu_qat.py`, read the steps(including QAT, model testing, model converting and xmodel dumping) in the script and run the step you want.
  ```shell
  bash code/run_qat.sh
  ```
   
   
### License

Apache License 2.0

For details, please refer to **[Vitis-AI License](https://github.com/Xilinx/Vitis-AI/blob/master/LICENSE)**


### Note

Data preprocess
```
  data channels order: BGR
  keeping the aspect ratio of H/W, resize image with bilinear interpolation to make the long side to be 416, pad the image with (114,114,114) along the height side to get image with shape of (H,W)=(416,416)
  ``` 


### Acknowledgement

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX.git)

