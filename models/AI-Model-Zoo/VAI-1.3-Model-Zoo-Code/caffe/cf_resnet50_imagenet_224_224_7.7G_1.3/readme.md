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
  # use python2
  make pycaffe
  ```
  Note: If you are in the released Docker env, there is no need to build Caffe.

### Preparation

1.Prepare datset.
  
  ImageNet dataset link: [ImageNet](http://image-net.org/download-images) 
  
  ```
  a.Users need to download the ImageNet dataset by themselves as it needs registration. The script of get_dataset.sh can not automatically download the dataset. 
  b.The downloaded ImageNet dataset needs to be organized as the following format:
    1) Create a folder of "data" along the "code" folder. Put the training data in +data/train and validation data in +data/validation.
    2) Data from each class for train or validation set needs to be put in a separate folder:
    +data/train
         +train/n01847000 
         +train/n02277742
         +train/n02808304
         +... 
    +data/validation
         +validation/n01847000 
         +validation/n02277742
         +validation/n02808304
         +... 
  ```
  
2.Preprocess dataset.

  ```shell
  cd code/gen_data
  # run get_dataset.sh and preprocess the dataset for model requirement.
  bash get_dataset.sh 
  ```
  
  ```
  The generated data directory structure after preprocessing is like as follows:
  +data/Imagenet/   
       +Imagenet/train_resize_256 #train images after using short side resize method.
       +Imagenet/val_resize_256 #val images after using short side resize method.  
       +Imagenet/train.txt #train gt file.
       +Imagenet/val.txt #val gt file.
  
  gt file is like as follows: 
    ILSVRC2012_val_00000001.JPEG 65
    ILSVRC2012_val_00000002.JPEG 970
    ILSVRC2012_val_00000003.JPEG 230
    ILSVRC2012_val_00000004.JPEG 809
    ILSVRC2012_val_00000005.JPEG 516
    
  # Users also can use their own scripts to preprocess the dataset as the above format.
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

3. Evaluate quantized caffemodel.
  ```shell
  # cd test of this model
  cd code/test
  # modify configure if you need, includes caffe root, model path, weight path... 
  bash quantized_test.sh
  ```

### Performance

|Acc |Float model performance on Imagenet| 
|----|----|
|Recall_1(%)|74.44|
|Recall_5(%)|91.85|

|Acc |Quantized(int8) model performance on Imagenet| 
|----|----|
|Recall_1(%)|73.35|
|Recall_5(%)|91.30|

### Model_info

1.data preprocess
  ```
  data channel order: BGR(0~255)                  
  resize: short side reisze to 256 and keep the aspect ratio.
  center crop: 224 * 224                            
  mean_value: 104, 107, 123
  scale: 1.0
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
      crop_size:224
      mean_value: 104
      mean_value: 107
      mean_value: 123
     }

    image_data_param {
      root_folder:"/path/to/images"
      source: "/path/to/quant.txt"
      #new_width: 224, Note: images should be resized firstly with short side reiszing to 256 and keeping the aspect ratio. 
      #new_height: 224
      batch_size: 16
    }
  }
  # quant.txt: image path label, if use relative image path here, please comment out above root_folder code
    000001.jpg 1
    000002.jpg 2
    000003.jpg 3

  ```
3.For quantization with finetuning mode: 
  ```
  use trainval.prototxt for model quantization.
  ```
