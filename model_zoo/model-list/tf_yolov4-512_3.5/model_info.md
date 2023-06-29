# YOLOv4

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
| Framework          | TensorFlow                              |
| Prune Ratio        | 0%                                      |
| FLOPs              | 91.2G                                   |
| Input Dims (H W C) | 512,512,3                               |
| FP32 Accuracy      | 0.487 mAP (0.50:0.95)                   |
| INT8 Accuracy      | 0.412 mAP (0.50:0.95)                   |
| Train Dataset      | COCO2017                                |
| Test Dataset       | COCO2017                                |
| Supported Platform | GPU, VEK280, V70                        |
  

### Paper and Architecture 

1. Network Architecture: YOLOv4

2. Paper Link: https://arxiv.org/abs/2004.10934

   
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

3. Download the official darknet weights and convert to tensorflow weights of .pb format

   You can download YOLOv4-Leaky darknet weights from https://github.com/AlexeyAB/darknet/wiki/YOLOv4-model-zoo. There are two versions we supported, YOLOv4-Leaky with input resolution 416x416 and YOLOv4-Leaky with input resolution 512x512.
   For input size 512, You can get the two files: yolov4-leaky.weights and yolov4-leaky.cfg.
   ```
   cd code
   python tools/model_converter/convert.py --yolo4_reorder --fixed_input_shape yolov4-leaky.cfg yolov4-leaky.weights yolov4-leaky.h5
   python tools/model_converter/keras_to_tensorflow.py --input_model yolov4-leaky.h5 --output_model yolov4-leaky.pb
   ```
   For input size 416, You can get the two files: yolov4-leaky-416.weights and yolov4-leaky-416.cfg.
   ```
   cd code
   python tools/model_converter/convert.py --yolo4_reorder --fixed_input_shape yolov4-leaky-416.cfg yolov4-leaky-416.weights yolov4-leaky-416.h5
   python tools/model_converter/keras_to_tensorflow.py --input_model yolov4-leaky-416.h5 --output_model yolov4-leaky-416.pb
   ```


### Use Guide

1. Evaluation
    Configure the model path and data path in [code/test/run_eval.sh](code/test/run_eval.sh)
    For input size 512. 
    ```shell
    # run the script under 'code/test'
    bash run_eval.sh
    ```
    For input size 416. 
    ```shell
    # run the script under 'code/test'
    bash run_eval_416.sh
    ```

### Quantize

1. Quantize tool installation

   Please refer to [vai_q_tensorflow](../../../src/vai_quantizer/vai_q_tensorflow1.x)
  
2. Quantize workspace

   You could use code/quantize/ folder.

   
### License

Apache License 2.0

For details, please refer to **[Vitis-AI License](https://github.com/Xilinx/Vitis-AI/blob/master/LICENSE)**


### Note

 1. Data preprocess
  ```
  data channel order: RGB(0~255)
  resize: keep aspect ratio of the raw image and resize it to target size
  input = input / 255
  ``` 

2. Node information
  ```
  input node: 'graph/image_input:0'
  output nodes: 'graph/conv2d_109/BiasAdd:0', 'graph/conv2d_101/BiasAdd:0', 'graph/conv2d_93/BiasAdd:0'
  ```

