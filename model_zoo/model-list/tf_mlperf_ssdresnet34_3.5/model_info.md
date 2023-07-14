# MLPerf SSD-ResNet34


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
| FLOPs              | 433G                                    |
| Input Dims (H W C) | 1200,1200,3                             |
| FP32 Accuracy      | 0.225 mAP                               |
| INT8 Accuracy      | 0.213 mAP                               |
| Train Dataset      | COCO2017                                |
| Test Dataset       | COCO2017                                |
| Supported Platform | GPU, VEK280, V70                        |
  

### Paper and Architecture 

Network Architecture: SSD-ResNet34
  
### Dataset Preparation

1. Automatically create train/val tfrecords dataset and test dataset. 
   
   ```
   cd code/gen_data
    
   # download coco2017 dataset
   # run get_dataset.sh, then will generate required dataset(tfrecords, images) for model.

   bash get_dataset.sh 
   ```

2. Prepare train/val tfrecord dataset(by manual).
  ```
  1. download coco2017 datatset and put them in data file holder.
  2. reorg coco as below: 
       data/coco2017/
                    |->train2017/
                    |->val2017/
                    |->instances_train2017.json
                    |->instances_val2017.json
  3. convert dataset from coco style to voc style.
     # config parameter of code/train/dataset/convert_coco2voc_like.py, more details can be found in args_parser of convert_coco2voc_like.py. 
     python convert_coco2voc_like.py
     outputs: 
       data/coco2017/
                    |->Annotations/
                    |->Images/
                    |->train2017.txt
                    |->val2017.txt
  4. generate tfrecords.
     # config parameter of code/train/dataset/convert_tfrecords.py, more details can be found in args_parser in convert_tfrecords.py
     python convert_tfrecords.py

  ```
3. Prepare test dataset for pb model(by manual).
  ```
  1. prepare test image(coco2017 val), and rename data/Images.
  ```  



### Use Guide

1. Train with tfrecords.
  ```shell
  cd code/train

  bash train.sh
  ```
2. Eval with tfrecords and ckpt.
  ```shell
  cd code/train

  python eval_ssd_large.py
  ``` 
3. Freeze pb with ckpt.

  ```shell
  cd code/train
  ## configure ckpt path in run_freeze_graph.
  bash run_freeze_graph.sh 
  ```
4. Evaluate pb model.
  ```shell
  cd code/test

  bash test.sh
  ```
 
 
### License

Apache License 2.0

For details, please refer to **[Vitis-AI License](https://github.com/Xilinx/Vitis-AI/blob/master/LICENSE)**


### Note

1. Data preprocess
  ```
  data channel order: RGB(0~255)                  
  input = input / 255
  resize: 1200 * 1200(BILINEAR) 
  std = [0.229, 0.224, 0.225]
  mean = [0.485, 0.456, 0.406]
  input = (input - mean) / std
  ``` 
  
2. Node information
  ```
  The name of input node: 'image:0'
  The name of output node: 'detection_bboxes:0, detection_scores:0, detection_classes:0, ssd1200/py_cls_pred:0, ssd1200/py_location_pred:0"
  ```
  
3. Quantize tool installation

   Please refer to [vai_q_tensorflow](../../../src/vai_quantizer/vai_q_tensorflow1.x)
  
4. Quantize workspace

   You could use code/quantize/ folder.
  
   
   