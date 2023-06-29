# OFA-YOLOv5


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
   - Searched by OFA (once-for-all) 
   
   
### Specification

| Metric             | Value                                   |
| :----------------- | :-------------------------------------- |
| Framework          | PyTorch                                 |
| Prune Ratio        | 0%                                      |
| FLOPs              | 48.88G                                  |
| Input Dims (H W C) | 640,640,3                               |
| FP32 Accuracy      | 0.436 mAP                               |
| INT8 Accuracy      | 0.421 mAP                               |
| Train Dataset      | COCO2017                                |
| Test Dataset       | COCO2017                                |
| Supported Platform | GPU, VEK280, V70                        |
  

### Paper and Architecture 

Network Architecture: YOLOv5
 
  
### Dataset Preparation

1. Dataset description
   - download COCO2017 dataset.(refer to this repo https://github.com/ultralytics/yolov5)

2. Dataset diretory structure
   ```
   + data/coco
     + annotations
     + images
     + labels
     + train2017.txt
     + val2017.txt
   ```


### Use Guide

1. Test and Evaluation
    ```shell
    cd code/

    bash run_test.sh
    ```
2. Quantize model

    ```shell
    cd code/

    bash run_quant.sh
    ```

3. QAT Training(quantizing model directly would lead large accuracy drop, so we provide quantization training scripts to improve quantization accuracy.)
   ```shell
   cd code/

   bash run_qat.sh
   ```
4. QAT Testing and Dump xmodel
   ```shell
   cd code/

   bash run_qat_test.sh
   ```
   
   
### License

Apache License 2.0

For details, please refer to **[Vitis-AI License](https://github.com/Xilinx/Vitis-AI/blob/master/LICENSE)**


### Note

 Data preprocess
  ```
  data channel order: RGB(0~255)
  input size: h * w = 640 * 640
  normalize: 1.0 / 255
  mean = (0.0, 0.0, 0.0)
  std = (1.0, 1.0, 1.0)
  input = (input * normalize  - mean) / std
  ```

