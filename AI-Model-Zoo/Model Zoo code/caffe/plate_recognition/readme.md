### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Performance](#performance)
5. [Model_info](#model_info)  

### Installation
1. Get xilinx caffe-xilinx code. 
  ```shell
  cd caffe-xilinx
  ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```shell
  # Modify Makefile.config according to your Caffe installation.
  cp Makefile.config.example Makefile.config
  make -j8
  # python2
  make pycaffe
  ```

### Preparation

1. prepare dataset.
  ```
  dataset includes image file and groundtruth file.
  image file: put train/val images.
  groundtruth file: format as "imagename plate_numbers plate_color"
  image resize: resize 96*288(H*W).
  
  generate two lmdb:
  install lmdb: pip install lmdb.
  a.generate image lmdb using fake label；"imagenmae 0"
  b.generate label lmdb using groundtruth: "imagename plate_numbers plate_color" with code/gen_data/createLableLMDB.py
  ```

### Train/Eval
1. Train your model and evaluate the model on the fly.
  ```shell
  # cd train of this model.
  cd code/train
  # modify configure if you need, includs caffe root, solver configure.
  bash train.sh 
  ```

2. Evaluate caffemodel.
  ```shell
  # cd test of this model
  cd test
  # modify configure if you need, includes caffe root, model path, weight path... 
  python test.py
  ```

### Performance

|Acc |Eval on dataset| 
|----|----|
|plate number(%)|99.51|
|plate color(%)|100|


### Model_info

1.data preprocess
```
1. data channel order: BGR(0~255)                  
2. resize: 96*288(h*w).
3. mean_value: 104, 117, 123
4. scale: 1.0
```
2.For quantization with calibration mode:
  ```
  Modify datalayer of test.prototxt for model quantization:
  a. Replace the "Input" data layer of test.prototxt with the "ImageData" data layer.
  b. Modify the "ImageData" layer parameters according to the date preprocess information.
  c. Provide a "quant.txt" file, including image path and label information with fake value(like 1).
  d. Give examples of data layer and "quant.txt":

  # data layer example
    layer {
    name: "data"
    type: "ImageData"
    top: "data"
    top: "label"
    include {
      phase: TRAIN
    }
    transform_param {
      mirror: false
      mean_value: 104
      mean_value: 117
      mean_value: 123
     }

    image_data_param {
      source: "quant.txt"
      new_width: 288 
      new_height: 96
      batch_size: 16
    }
  }
  # quant.txt: image path label
    images/000001.jpg 1
    images/000002.jpg 1
    images/000003.jpg 1

  ```
3.For quantization with finetuning mode: 
  ```
  use trainval.prototxt for model quantization.
  ```
