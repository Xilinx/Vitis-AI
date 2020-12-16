### Contents
1. [Installation](#installation)
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
Note: If you are in the released Docker env, there is no need to build Caffe.

### Demo
 run demo
  ```shell
  # modify the "cafde_xilinx_dir" in code/test/demo.sh
  float: sh code/test/demo.sh
  quantized: sh code/test/demo_quantized.sh
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
  float: sh code/test/demo.sh
  quantized: sh code/test/demo_quantized.sh
  ```
3. Evaluate.
  ```shell
  # Evaluate mAP
  python code/test/evaluation.py -gt_file code/test/gt_detection.txt -result_file code/test/result.txt
  ```
### Performance
   ```shell
   Test images: vmss 10026
   Model: tiny_yolov3
   Classes: 10
   float mAP: 97.391% 
   Quantized mAP: 96.5% 
   ```
### Model info
```
1. data channel order: RGB(0~255)
2. resize: keep aspect ratio of the raw image and resize it to make the length of one side firstly equal to 416 
3. input = input / 256
4. padding: to generate the input image with size=416x416(H * W), pad the scaled image with 0.5
```
