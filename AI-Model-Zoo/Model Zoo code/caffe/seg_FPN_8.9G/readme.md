### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Demo](#demo)
4. [Train/Eval](#traineval)
5. [Performance](#performance)
6. [Model_info](#model_info)

### Installation
1. Get caffe-xilinx code.
  ```shell
  unzip caffe-xilinx.zip
  cd caffe-xilinx
  ```

2. Build the code. Please follow [caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```shell
  # Modify Makefile.config according to your caffe installation.
  cp Makefile.config.example Makefile.config
  make -j
  make pycaffe
  ```

### Preparation

1. dataset describle.
  ```
  dataset includes image file and groundtruth file.
  image file: put train/val images.
  groundtruth file: put train/val labels
  ```
2. prepare datset.

  ```shell
  cd code/

  # check dataset soft link or user download cityscapes dataset (https://www.cityscapes-dataset.com/downloads)
  # grundtruth folder: gtFine_trainvaltest.zip [241MB]
  # image folder: leftImg8bit_trainvaltest.zip [11GB]
  # put the grundtruth folder and image folder in  `data` directory.
  # run get_dataset.sh, then will generate required dataset for model.

  bash get_data.sh 

  * `data` directory structure like:
     + cityscapes
       + leftImg8bit
         + train
         + val
       + gtFine
         + train
         + val
       + lists
         + train_img_seg.txt
         + val_img_seg.txt

  ```

### Train/Eval
1. Train your model and evaluate the model on the fly.
  ```shell
  # cd train of this model.
  # modify configure if you need, includs caffe root, solver configure.
  cd ./code/train/
  bash train.sh
  ```

2. Evaluate caffemodel.
  ```shell
  # modify configure including caffe path, test_prototxt path, weight path, test image path... 
  cd ./code/test
  python test.py
  # if final mIoU with float model is 56.69% then test sucessfully, or fail.
  ```

### Demo
  ```shell
  # modify configure including caffe path, test_prototxt path, weight path 
  cd ./code/demo
  python demo.py
  ```

### Performance

|mIoU | Eval on Cityscapes | 
|---- |----|
|256x512|56.69%|


### Model_info

1. data preprocess
```
1.1). data channel order: BGR(0~255)
1.2). resize: 256 * 512(H*W)
1.3). mean_value: 104, 117, 123
1.4). scale: 1.0
```

2. quantize the network with calibration mode
```
2.1) Replace the "Input" layer of the "test.prototxt" file with the "ImageData" data layer.
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
    new_width: 512
    new_height: 256
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
3. quantize the network with finetuning mode
```
3.1) Using the same data layer, loss layer and evaluation layer with "trainval.prototxt"

```