### Contents
1. [Installation](#installation)
2. [Demo](#Demo)
3. [Preparation](#Preparation)
4. [Train/Eval](#traineval)
5. [Performance](#Performance)
6. [Model info](#Model info)

### Installation
1. Get the code. We will call the directory that you cloned Caffe into `$CAFFE_ROOT`
  **Note:** To download Caffe_Xlinx
  
  ```shell
  unzip caffe-xilinx.zip
  cd caffe-xilinx
  ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```shell
  # Modify Makefile.config according to your Caffe installation.
  # Modify "ori_flag,label_ori,label_with_ori" parameter from "false" to "true".
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
  # modify the "caffe_xilinx_dir" in code/test/demo.sh
  float: sh code/test/demo.sh
  quantized: sh code/test/demo_quantized.sh
  ```
### Preparation
1. Dataset Diretory Structure like:
   ```shell
   + train
     + images
        + images_id1.jpg
        + iamges_id2.jpg
     + label_txt
        + images_id1.txt
        + images_id2.txt
     + seg_label
        + images_id1.png
        + iamges_id2.png
      
     label.txt: please refer code/train/label_txt/*.txt
        image_name label_1 xmin1 ymin1 xmax1 ymax1
        image_name label_2 xmin2 ymin2 xmax2 ymax2
        ...
     ```

2. Create the LMDB 
   ```shell
    Because the data is private so you should copy your data to 'code/train/images/' and modify the 'train.txt' and 'test.txt' information. You can use the 'create_data.sh.' convert the data to IMDB.
   #modify the "caffe_xilinx_dir" in "create_data.sh"
   sh ./code/train/create_data.sh
   ```

### Train/Eval
1. train your model
   ```shell
   sh ./code/train/trainval.sh
   ```
2. Evaluate the most recent snapshot.
   ```shell
   # If you would like to test a model you trained, you can do:
   # 1. add the all images path to  images.txt and  run 
   # 2. modify the threshlod 0.3 to 0.005 
   float: sh code/test/demo.sh
   quantized: sh code/test/demo_quantized.sh
   # Segmetation result stored in the "seg_result" folder  
   # Detection results stored in the "./code/test/result.txt"
   ```
   Evaluate mIou and mAP.
   ```shell
   # Evaluate mAP
   # Ensure that the image name is consistent
   python code/test/evaluation_det.py  -gt_file code/test/gt_labels.txt -result_file code/test/result.txt
   # Evaluate the mIou , The classes value default is 16.
   python code/test/evaluation_seg.py seg PATH_TO_GT_FOLDER seg_result
   ```
### Performance
   ```shell
   Test images: bdd100 val 988
   Model: ssd-resnet18
   Classes-detection: 8
   Classes-segmentation: 16  
   Float model: mAP: 22.28% &  mIou: 40.88%
   Quantized model: mAP: 21.4% &  mIou: 40.58%
   ```
### Model info
1. data preprocess
```
1. data channel order: BGR(0~255)                  
2. resize: 288 * 512(H * W) 
3. mean_value: 104, 117, 123
4. scale: 1
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
      mean_value: 104
      mean_value: 117
      mean_value: 123
     }

    image_data_param {
      source: "quant.txt"
      new_width: 512  
      new_height: 288
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
