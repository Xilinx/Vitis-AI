### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [demo](#demo)
4. [Train/Eval](#traineval)
5. [Performance](#performance)
6. [Model_info](#model_info)

### Installation
1. Get Xilinx caffe-xilinx code.
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
3. opencv-python
  ```shell
  #You may need to make sure opencv-python is installed
  pip install opencv-python
  ```

### Preparation

1. prepare train dataset.
   
  train dataset: [WIDER Face](http://shuoyang1213.me/WIDERFACE/index.html)
  ```
  1.please download and ertract WIDER_train to directory data/train/
  2.please download and ertract Face annotations, then put wider_face_train_bbx_gt.txt to directory data/train/WIDER_train/
  3.Dataset Directory Structure like:
    + data/train
        + WIDER_train
            + images
            + wider_face_train_bbx_gt.txt
  ```
  ```shell
  cd code/gen_data/
  sh gen_train_lmdb.sh
  ```

2. prepare test dataset.
   
  test dataset: [FDDB](http://vis-www.cs.umass.edu/fddb/index.html)
  
  ```shell
  cd code/gen_data/
  sh gen_testdata_fddb.sh
  ```

#### demo

1. run demo
  ```shell
  cd code/test/visualTest/
  sh demo.sh
  ```

### Train/Eval
1. Train your model.
  ```shell
  cd code/train
  sh train.sh 
  ```

2. Evaluate your model.
  
  FDDB testing requires the third-party evaluation tools, please download and compile it.
  You can refer to [evaluation tool](http://vis-www.cs.umass.edu/fddb/results.html)
  ```shell
  cd code/test/precisionTest/
  sh test.sh
  #if recall rate is 0.8833 for float model, test sucessfully.
  ```
 
### Performance

|precision |Eval on FDDB| 
|----|----|
|Recall(%)|88.33@fp=100|

### Model_info

data preprocess
  ```
  data channel order: BGR(0~255)
  padding: padding to image according to the ratio of 360:640(H:W)                  
  resize: 320 * 320
  mean_value: 128, 128, 128
  scale: 1.0
  ```

Quantize the network with calibration mode
1. Replace the "Input" layer of the test.prototxt file with the "ImageData" data layer.
2. Provide a "quant.txt" file, including image path and label information with fake value(like 1).
3. Give examples of data layer and "quant.txt":

```shell
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
    batch_size: 32
    new_width: 320
    new_height: 320
  }
  transform_param {
    mirror: false
    mean_value: 128
    mean_value: 128
    mean_value: 128
    scale: 1
  }
}
```
```
# quant.txt: image path label
images/000001.jpg 1
images/000002.jpg 2
images/000003.jpg 3
...
```
