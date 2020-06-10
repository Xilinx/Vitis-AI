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
### demo
  ```shell
  cd code/test
  # config caffe-root, prototxt, weights if want to modify path, and more details of parameter can be saw in test.py.
  python demo.py
  # res_output.jpg can be found in output.

  ```

### Preparation data
1. Auto create train/val lmdb dataset and test dataset. 
  ```shell
  cd code/gen_data

  bash get_dataset.sh
  ``` 
2. Prepare train/val lmdb dataset(manul).
  ```
  1. download coco2014 datatset.
  2. reorg coco as below: 
       coco2014/
             |->train2014/
             |->val2014/
             |->instances_train2014.json
             |->instances_val2014.json
    # config parameter of code/gen_data/convert_coco2voc_like.py, more details can be found in args_parser of convert_coco2voc_like.py. 
  3. python convert_coco2voc_like.py
     outputs: 
       coco2014/
             |->Annotations/
             |->Images/
             |->train2014.txt
             |->val2014.txt
  4. generate lmdb and test_name_size.txt
     # config parameter of code/gen_data/create_data.py, more details can be found in args_parser in create_data.py
     python create_data.py

  ```
3. prepare test dataset(manul)
  ```
  1. prepare test image(coco2014 val) and image_list file(which only record imagename without postfix).
  2. prepare voc-like annotations file generated in above step3.
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
  # configure parameter of test/test.py, more details can be found in args_parser in test.py.
  python test.py
  ```
3. Note that.
   data source path of prototxt should be modified as user's
 
### Performance

|metric |Eval on coco2014 val| 
|----|----|
|11 point map(%)|59.0264|

### Model_info

1. data preprocess
```
 1. data channel order: BGR(0~255)                  
 2. resize: 360 * 640(H * W) 
 3. mean_value: 104, 117, 123
 4. scale: 1
```

2.For quantization with calibration mode:
  ```
  Modify datalayer of test.prototxt for model quantization:
  a. Replace the "Input" data layer of test.prototxt with the "ImageData" data layer.
  b. Modify the "ImageData" layer parameters according to the date preprocess information.
  c. Provide a "quant.txt" file, including image path and label information with fake value(like 1).
  d. Give examples of data layer and "quant.txt":
  note: mbox_priorbox layer need be ignored while quantizing the model.

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
      mean_value: 104
      mean_value: 117
      mean_value: 123
     }

    image_data_param {
      source: "quant.txt"
      new_width: 640  
      new_height: 360
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
