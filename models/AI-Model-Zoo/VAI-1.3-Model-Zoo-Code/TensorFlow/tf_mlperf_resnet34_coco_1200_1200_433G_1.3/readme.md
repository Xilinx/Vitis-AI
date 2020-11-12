### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Performance](#performance)
5. [Model_info](#model_info)
6. [Quantize](#quantize)

### Installation
1. Environment requirement 
   tensorflow 1.15
   
2. Installation
   - Create virtual envrionment and activate it:
   ```shell
   conda create -n tf_ssd python=3.6
   conda activate tf_ssd
   ```
   - Install all the python dependencies using pip:
   ```shell
   pip install -r requirements.txt
   ```

### Preparation

1. dataset describe.
   ```
   reorg coco as below:(get_dataset.sh done) 
   data/coco2017/
             |->train2017/
             |->val2017/
             |->instances_train2017.json
             |->instances_val2017.json
   ```
2. prepare testdatset.
   
   ```
   cd code/gen_data
    
   # download coco dataset
   # run get_dataset.sh, then will generate required dataset for model.

   bash get_dataset.sh 
   ```

### Train/Eval
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

### performance
|Acc |Pb on coco2017 val|
|----|----|
|mMAP(%)|22.5|


### Model_info

1.  data preprocess
  ```
  data channel order: RGB(0~255)                  
  input = input / 255
  resize: 1200 * 1200(BILINEAR) 
  std = [0.229, 0.224, 0.225]
  mean = [0.485, 0.456, 0.406]
  input = (input - mean) / std
  ``` 
2. node information
  ```
  The name of input node: 'image:0'
  The name of output node: 'detection_bboxes:0, detection_scores:0, detection_classes:0, ssd1200/py_cls_pred:0, ssd1200/py_location_pred:0"
  ```

### Quantize
1- Quantize tool installation
  See [vai_q_tensorflow](https://github.com/Xilinx/Vitis-AI/tree/master/Vitis-AI-Quantizer/vai_q_tensorflow)

2- Quantize workspace
  See [quantize](./code/quantize/)
