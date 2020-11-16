# Endoscopy Diseases Segmentation with FPN-ResNet18
### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Eval](#eval)
4. [Performance](#performance)
5. [Model_info](#model_info)

### Installation
1. Get caffe-xilinx code.
  ```shell
  unzip caffe-xilinx.zip
  cd caffe-xilinx
  ```

2. 2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```shell
  # Modify Makefile.config according to your caffe installation.
  cp Makefile.config.example Makefile.config
  make -j
  make pycaffe
  ```
  Note: If you are in the released Docker env, there is no need to build Caffe.

### Preparation

1. dataset describle.
  ```
  download EDD2020 dataset.
  dataset includes image file, groundtruth file and a validation image list file.
  image file: put to data/images.
  groundtruth file: put to data/labels
  ```
2. prepare datset.

  ```shell
  cd code/

  # please download EDD2020 dataset (https://edd2020.grand-challenge.org/Home/).
  # put the grundtruth folder and image folder in  `data` directory.
  # We split the annotated dataset into training/validation sets.
  * `data` directory structure like:
     + EDD
       + images
       + labels
       + val_img_seg.txt
  ```

### Eval

1. Evaluate caffemodel.
  ```shell
  # modify configure if you need, includes caffe path, model path, weight path, data path...
  bash eval.sh
  ```

2. Evaluate quantized caffemodel
  ```shell
  # modify configure if you need, includes caffe path, quantized model path, quantized weight path, data path...
  bash test_quantized.sh
  ```
  
### Performance

|Input | FLOPs | Performance on EDD2020 | 
|---- |----|----|
|320x320|45.3G| mean dice=0.8203; mean jaccard=0.7925; F2-score=0.8075|

|Input | FLOPs | INT8 Performance on EDD2020 | 
|---- |----|----|
|320x320|45.3G| mean dice=0.8055; mean jaccard= 0.7772; F2-score=0.7925|

### Model_info

1. data preprocess
```
1.1). data channel order: BGR(0~255)
1.2). resize: 320x320(H*W)
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
    new_width: 320
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