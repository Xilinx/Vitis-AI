### Contents
1. [Installation](#installation)
2. [demo](#demo)
3. [Preparation](#preparation)
4. [Train/Eval](#traineval)
5. [Performance](#performance)
6. [Model_info](#model_info)  


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
3. Install pycocotools for coco evaluation.
  ```shell
  pip install pycocotools
  ```

### demo
  ```shell
  cd code/test
  # config caffe-root, prototxt, weights if want to modify path. and image list for demo is in data/coco/demo.txt
  bash demo.sh
  # results can be found in demo.

  ```

### Preparation data
1. Auto create test dataset. 
  ```shell
  cd code/gen_data

  bash get_dataset.sh
  ``` 
2. Prepare test dataset(manul).
  ```
  1. download coco2014 val datatset.
  2. reorg coco as below: 
       data/coco2014/
                    |->val2014/
  ```
3. Note that coco2014 val-5k is used for evaluation, which is split as yolov4 offical repo does. 
   And image list can be found in data/coco/image.txt

## Train/Eval

1. Evaluate caffemodel.
  ```shell
  # cd test of this model
  cd code/test
  # configure parameter of test/test.sh.
  bash test.sh
  ```

2. Evaluate quantized caffemodel.
  ```shell
  # cd test of this model
  cd code/test
  # configure parameter of test/quantized_test.sh.
  bash quantized_test.sh
  ```
 
### Performance

|metric |Eval on coco2014-5k val| 
|----|----|
|mmap(%)|39.5|

### Model_info

1. data preprocess
```
 1. data channel order: BGR(0~255)                  
 2. resize: 416 * 416(H * W) 
 3. mean_value: 0.0, 0.0, 0.0
 4. scale: 1 / 255.0
 5. reisze mode: biliner
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
    image_data_param {
      root_folder:"/path/to/images"
      source: "quant.txt"
      batch_size: 16
    }
    transform_param {
      mirror: false
      yolo_width: 416
      yolo_height: 416
    }
  }
  # quant.txt: image path label
    000001.jpg 1
    000002.jpg 1
    000003.jpg 1

  ```
3.Note that using "-sigmoided_layers" parameter with output layers "layer138-conv,layer149-conv,layer160-conv" while quantizing yolov4 model would lead better performance.
