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
  Note: If you are in the released Docker env, there is no need to build Caffe.

### Preparation

1. Prepare dataset.
  ```  
  a. dataset includes images and groundtruth file.
    1) groundtruth file format as following:
       imagename: 123342315_京DH09R8_20171201160736.jpg
       number of plates: 1
       coordinates of plate(clockwise order): 204 271 258 270 258 286 204 288

       case:
       123342315_京DH09R8_20171201160736.jpg
       1
       204 271 258 270 258 286 204 288
       123342327_京AJ1620_20171201160732.jpg
       1
       44 275 82 276 85 291 44 287
       123362928_粤SL1J70_20171201164354.jpg
       1
       66 178 114 180 113 199 66 196

      Note that image must be resized to 320 * 320 firstly and coordinates of groundtruth file must be adapted to 320*320 also. 

  b. dataset structure:
     data/
         + train_images/
         + train_gt.txt
  ```
  b. generate lmdb command:
  ```
  caffe-xilinx/build/tools/convert_direct_txt "path/to/train_images/" "path/to/train_gt.txt" train_plate_landmark_lmdb
  # for vitis-ai docker env, use:
  # /opt/vitis_ai/conda/envs/vitis-ai-caffe/bin/convert_direct_txt "path/to/train_images/" "path/to/train_gt.txt" train_plate_landmark_lmdb
  

  ```
  finally move `train_plate_landmark_lmdb` to `data`.

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

2. Evaluate quantized caffemodel.
  ```shell
  # cd test of this model
  cd code/test
  # modify configure if you need, includes caffe root, model path, weight path... 
  bash quantized_test.sh
  ```


### Performance

Evaluate on private dataset with 4054 images.  

|Acc@iou0.5 |Float model performance on dataset| 
|----|----|
|Recall_1(%)|96.60|


|Acc@iou0.5 |Quantized(int8) model performance on dataset| 
|----|----|
|Recall_1(%)|96.50|


### Model_info

1. data preprocess
```
1. data channel order: BGR(0~255)                  
2. resize: 320*320(h*w).
3. mean_value: 128, 128, 128
4. scale: 1.0
```
2.For quantization with calibration mode:
  ```
  Modify datalayer of test.prototxt for model quantization:
  a. Replace the "Input" data layer of test.prototxt with the "ImageData" data layer.
  b. Modify the "ImageData" layer parameters according to the data preprocess information.
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
      mean_value: 128
      mean_value: 128
      mean_value: 128
     }

    image_data_param {
      source: "quant.txt"
      new_width: 320 
      new_height: 320
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
