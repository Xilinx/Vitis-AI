### Contents
1. [Installation](#installation)
2. [Preparation](#Preparation)
2. [Demo](#Demo)
3. [Train/Eval](#traineval)
4. [Performance](#Performance)
5. [Model info](#Model info)

### Installation
1. Get the code. We will call the directory that you cloned Caffe into `$CAFFE_ROOT`
  **Note:** To download caffe-xilinx
  
  ```shell
  unzip caffe-xilinx.zip
  cd caffe-xilinx
  ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```shell
  # Modify Makefile.config according to your Caffe installation.
  cp Makefile.config.example Makefile.config
  make -j8
  # Make sure to include $CAFFE_ROOT/python to your PYTHONPATH.
  # python version(python2)
  make py
  ```
### Preparation
   Download trainval and test dataset. you can run below the command get dataset. By default, we assume the data is stored in `data/`
  ```Shell
  # Download the data.
  cd data/
  sh download_and_preprocess.sh
  ```

### Demo
 run demo
  ```shell
  # modify the "cafde_xilinx_dir" in code/test/demo.sh
  sh code/test/demo.sh
  ```

### Train/Eval
1. The Yolo models are trained on the darknet, so we do not provide training code. If you want to deploy your trained darknet models, we offer conversion tools, and it can convert darknet models into caffe models.
  ```shell
      export PYTHONPATH='caffe_xilinx_dir/python/'
      python scripts/convert.py
  ```
2. Evaluate the models we provide.
  ```shell
  # If you would like to test a model you trained, you can do:
  sh code/test/demo.sh
  ```
3. Evaluate.
  ```shell
  # Evaluate mAP
  python code/test/evaluation.py -gt_file code/test/gt_detection.txt -result_file code/test/result.txt
  ```
### Performance
  ```shell
   Test images: VOC2007 4952
   Model: yolov2
   Classes: 20
   mAP: 75.76% 
   ```
### Model info
1. data info
```
1.1) data channel order: RGB(0~255)
1.2) resize: keep aspect ratio of the raw image and resize it to make the length of the longer side equal to 448
1.3) input = input / 256
1.4) padding: to generate the input image with size=448x448, pad the scaled image with 0.5
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
    batch_size: 16
  }
  transform_param {
    mirror: false
    yolo_width: 448
    yolo_height: 448
  }
}

# quant.txt: image_path label
images/000001.jpg 1
images/000002.jpg 1
images/000003.jpg 1
...
```
