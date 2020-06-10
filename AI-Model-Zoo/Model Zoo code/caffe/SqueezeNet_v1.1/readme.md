### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Performance](#performance)
5. [Model_info](#model_info)

### Installation
1. Get xilinx caffe-xilinx code.
  ```shell
  unzip caffe-xilinx.zip
  cd caffe-xilinx
  ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```shell
  # Modify Makefile.config according to your Caffe installation.
  cp Makefile.config.example Makefile.config
  make -j8
  make pycaffe
  ```

### Preparation

1. dataset describe.
  ```
  dataset includes image file and groundtruth file.
  image file: put train/val images.
  groundtruth file: format as "imagename label"
  ```
2. prepare datset.
  ```shell
  cd code/gen_data

  # check dataset soft link or 
  # user download train and validation dataset, rename them with "train" and "validation".

  # dataset put or generated in data fileholder.
  # dataset fortmat: A big file(validation) includes 1000 subfiles, echo subfile put same class images.  
  # run get_dataset.sh, then will generate required dataset for model.

  bash get_dataset.sh 
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
  cd code/test
  # modify configure if you need, includes caffe root, model path, weight path... 
  bash test.sh
  ```

### Performance

|Acc |Eval on Imagenet| 
|----|----|
|Recall_1(%)|54.10|
|Recall_5(%)|78.01|


### Model_info

1.data preprocess
  ```
  data channel order: BGR(0~255)                  
  resize: short side reisze to 256 and keep the aspect ratio.
  center crop: 227 * 227                           
  mean_value: 104, 117, 123
  scale: 1.0
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
      crop_size:227
      mean_value: 104
      mean_value: 117
      mean_value: 123
     }

    image_data_param {
      source: "quant.txt"
      #new_width: 227, Note: images should be resized firstly with short side reiszing to 256 and keeping the aspect ratio. 
      #new_height: 227
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
