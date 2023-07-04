# YOLOv3


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
| Framework          | TensorFlow2                             |
| Prune Ratio        | 0%                                      |
| FLOPs              | 65.9G                                   |
| Input Dims (H W C) | 416,416,3                               |
| FP32 Accuracy      | 0.377 mAP (0.50:0.95)                   |
| INT8 Accuracy      | 0.331 mAP (0.50:0.95)                   |
| Train Dataset      | COCO2017                                |
| Test Dataset       | COCO2017                                |
| Supported Platform | GPU, VEK280, V70                        |
  

### Paper and Architecture 

1. Network Architecture: YOLOv3

2. Paper Link: https://arxiv.org/abs/1804.02767

   
### Dataset Preparation

1. Dataset description

The dataset for evaluation is MSCOCO val2017 set which contains 5000 images.

2. Download and prepare the dataset

Run the script `prepare_data.sh` to download and prepare the dataset.
   ```shell
   bash code/test/download_data.sh
   bash code/test/convert_data.sh
   ```
Dataset diretory structure: 
   ```shell
   # val2017 and annotations are unpacked from the downloaded data
   + data
     + val2017
       + 000000000139.jpg
       + 000000000285.jpg
       + ...
     + annotations
       + instances_train2017.json
       + instances_val2017.json
       + ...
     + val2017.txt
   ```

3. Download the official darknet weights and convert to tensorflow weights of .h5 format
   ```
   cd code/test
   wget -O ../../float/yolov3.weights https://pjreddie.com/media/files/yolov3.weights
   python tools/model_converter/convert.py cfg/yolov3.cfg ../../float/yolov3.weights ../../float/yolov3.h5
   ```


### Use Guide

1. Evaluation
    Configure the model path and data path in [code/test/run_eval.sh](code/test/run_eval.sh)
    ```shell
    # run the script under 'code/test'
    bash run_eval.sh
    ```
  
   
### License

Apache License 2.0

For details, please refer to **[Vitis-AI License](https://github.com/Xilinx/Vitis-AI/blob/master/LICENSE)**


### Note

 1. Data preprocess
  ```
  data channel order: RGB(0~255)
  resize: keep aspect ratio of the raw image and resize it to make the length of the longer side equal to 416
  padding: pad along the short side with pixel value 128 to generate the input image with size = 416 x 416
  input = input / 255
  ``` 

2. Node information
  ```
  input node: 'image_input:0'
  output nodes: 'conv2d_58:0', 'conv2d_66:0', 'conv2d_74:0'
  ```

### Quantize

1. Quantize tool installation

   Please refer to [vai_q_tensorflow](../../../src/vai_quantizer/vai_q_tensorflow2.x)
  
2. Quantize workspace

   You could use code/quantize/ folder.
  


### Acknowledgement

[keras-YOLOv3-model-set](https://github.com/david8862/keras-YOLOv3-model-set.git)