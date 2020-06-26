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

1. Prepare test images
   
  You can randomly select several images from existing test dataset or use your own testset directly.
  ```shell
  cd code/test/visualTest
  mkdir testImages
  cp ../../../../data/test/images/2002/08/26/big/img_265.jpg ./testImages/
  find testImages/ -name "*.jpg" > image_list_test.txt  
  ```
2. run demo
  ```shell
  sh demo.sh
  # You can see visualization output results from ./output. 
  ```

### Train/Eval
1. Train your model.
  ```shell
  cd code/train
  sh train.sh 
  ```

2. Evaluate your model.
  
  FDDB testing requires the third-party evaluation tools, please download [evaluation tool](http://vis-www.cs.umass.edu/fddb/evaluation.tgz) to code/test/precisionTest/, unzip and compile it.

  If you have problems compiling it, you can refer to [FAQ](http://vis-www.cs.umass.edu/fddb/faq.html)

  ```shell
  cd code/test/precisionTest/
  sh test.sh # generate face detection result file FDDB_results.txt
  #use the evaluation tool to generate ROC files. You can try to run the following command.
  evaluation/evaluate -a $Path/data/test/FDDB_annotation.txt -d $Path/FDDB_results.txt -i $Path/data/test/images/ -l $Path/data/test/FDDB_list.txt -r $PATH_WORK_DIR 
  #Path is absolute path. 
  #if recall rate is 0.8931 @fp=100 for float model, test sucessfully.
  ```
 
### Performance

|precision |Eval on FDDB| 
|----|----|
|Recall(%)|89.31@fp=100|

### Model_info

data preprocess
  ```
  data channel order: BGR(0~255)
  padding: padding to image according to the ratio of 360:640(H:W)                  
  resize: 360 * 640
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
    new_width: 640
    new_height: 360
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
