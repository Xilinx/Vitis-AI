# Robot Instrument Segmentation using FPN_ResNet18
### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Eval](#eval)
4. [Performance](#performance)
5. [Model_info](#model_info)

### Installation
1. Get the code. We will call the directory that you cloned Caffe into `$CAFFE_ROOT`
  **Note:** To download caffe-linx
  
  ```shell
  unzip caffe-xilinx.zip
  cd caffe-xilinx
  ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```shell
  # Modify Makefile.config according to your Caffe installation.
  cp Makefile.config.example Makefile.config
  make -j
  # Make sure to include $CAFFE_ROOT/python to your PYTHONPATH.
  # python version(python2)
  make py
  ```
  Note: If you are in the released Docker env, there is no need to build Caffe.

### Preparation

1. dataset describle.
  ```
  dataset includes image file, groundtruth file and a validation image list file.
  ```
2. prepare datset.

  ```shell

  # please download EndoVis'15 dataset (Rigid Instruments Segmentation Traing/Testing zip files) (http://opencas.webarchiv.kit.edu/?q=node/30).
  # unzip and put the download dataset in  `data` directory.

  * `data` directory structure like:
     + data/Endov
       + train (total 160 images/masks)
         + OP1
           - Raw
           - Masks
         + OP2
         + OP3
         + OP4
       + val (total 140 images/masks)
         + OP1
         + OP2
         + OP3
         + OP4
         + OP5
         + OP6
  ```
  Note: We call the test set as validation set.
### Eval

1. Evaluate caffemodel.
  ```shell
  # modify configure if you need, includes caffe root, model path, weight path, data path...
  bash test.sh
  
  # Switch the Python2 environment to Python3 to do evaluation
  bash eval.sh
  ```
2. Evaluate quantized caffemodel.  
  ```shell
  # modify configure if you need, includes caffe root, quantized model path, quantized weight path, data path...
  bash test_quantized.sh
  
  # Switch the Python2 environment to Python3 to do evaluation, and modify the saved result path of the quantized model
  bash  eval_quantized.sh
  ```  
### Performance

|Input | FLOPs | Performance on EndoVis'15 Test Set | 
|---- |----|----|
|240x320|13.75G|Dice = 80.5%, Jaccard = 72.7%|

|Input | FLOPs | INT8 Performance on EndoVis'15 Test Set | 
|---- |----|----|
|240x320|13.75G|Dice = 80.1%, Jaccard = 72.3%|

### Model_info

1. data preprocess
```
1.1). data channel order: BGR(0~255)
1.2). resize: 240x320(H*W)
1.3). mean_value: 104, 117, 123
1.4). scale: 1.0
```
2. quantize the network with calibration mode
```
2.1) Replace the "Input" layer of the "pytorch2caffe_mergebn2conv.prototxt" file with the "ImageData" data layer.
2.2) Modify the "ImageData" layer parameters according to the data preprocess information.
2.3) Provide a "quant.txt" file, including image path and label information with fake value(like 1).
2.4) Give examples of data layer and "quant.txt"

# ImageData layer
layer {
  name: "data"
  type: "ImageData"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  image_data_param {
    source: "quant.txt"
    batch_size: 4
    new_width: 240
    new_height: 320
  }
  transform_param {
    mirror: false
    mean_value: 104
    mean_value: 117
    mean_value: 123
    scale: 1.0
  }
}

# quant.txt: image_path label
images/000001.jpg 1
images/000002.jpg 1
images/000003.jpg 1
...
```
