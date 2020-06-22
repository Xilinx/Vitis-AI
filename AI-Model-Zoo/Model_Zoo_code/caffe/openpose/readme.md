
# Openpose Model

### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Performance](#performance)
5. [Model_info](#model_info)

### Installation
1. Get xilinx caffe_xilinx code.
  ```shell
  unzip caffe_xilinx.zip
  cd caffe_xilinx
  ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```shell
  # Modify Makefile.config according to your Caffe installation.
  cp Makefile.config.example Makefile.config
  make -j8
  make pycaffe
  ```

### Preparation

1. dataset describle.
  ```
  dataset includes image file and groundtruth file.
  image file: put train/val images. the image files are jpeg file, the suffix should be '.jpg'
  groundtruth file: format as followings. the groundtruth file is a json file, the suffix should be '.json'
  ```
 
  * groundtruth file format:
 
  ```xml
  [
    {
        "image_id": "image_name", 
        "keypoint_annotations": 
        {
            "human1":[px1, py1, pt1, px2, py2, pt2, ..., px14, py14, pt14],
            "human2",:[...],
            ...,
            "humanN",:[...]
        },
        "human_annotations": 
        {
            "human1": [x1,y1,x2,y2],
            "human2": [x1,y1,x2,y2],
            ...,
            "humanN":[x1,y1,x2,y2]
        }
    },
    {
        ...
    }
  ]
  ```
  
in the `keypoint_annotation` there are 14 key points, which represents: 1: R_shoulder, 2: R_elbow, 3: R_wrist, 4: L_shoulder, 5: L_elbow, 6: L_wrist, 7: R_hip, 8: R_knee, 9: R_ankle, 10: L_hip, 11: L_knee, 12: L_ankle, 13: head, 14: neck, respectively. Each point has three elements: px, py, pt. (px, py) mean the coordinate of this point, pt means the type of this point. it has three values: 1,2,3. 1 means it is a visible point. 2 means it is an invisible point and 3 mean it is an invalid point.

2. preprocess datset. After you prepare your dataset, run the following code.
  ```shell
  python code/preprocess.py --anno train_anno_file_path --data train_image_data_path --output data/train
  python code/preprocess.py --anno validation_anno_file_path --data validation_image_data_path --output data/validation
  #it will generate two files meta.bin and path.txt, please move into a folder and modify source value to the folder's path in the datalayer of trainval.prototxt 
  ```

### Train/Eval
1. Train your model and evaluate the model on the fly.
  ```shell
  # modify configure if you need, includs caffe root, solver configure.
  bash code/train/train.sh 
  ```

2. Evaluate caffemodel.
  ```shell
  # cd test if this model
  # modify configure if you need, includes caffe root, model path, weight path... 
  python code/test/test.py --gpus 0,1,2,3 --data your_image_data_path --caffe $CAFFE_ROOT --weights your_caffemodel_file --model your_prototxt_file --anno your_annotation_file --input data
  ```


### Model info
   data preprocess information
  ```
   data channel order: BGR(0~255)                  
   resize: 388 * 388(H * W) 
   mean_value: 128, 128, 128
   scale: 1
  ```
   For quantization with calibration mode:
  ```
   1. Replace the original network file data layer with the "ImageData" data layer
   2. Modify the "ImageData" layer parameters according to the date preprocess information
   3. Provide a "quant.txt" file, including image path and label information, but the label can randomly give some values
   4. Give examples of data layer and "quant.txt":

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
       new_width: 368
       new_height: 368
       batch_size: 16
     }
   }
   # quant.txt: image path label
     images/000001.jpg 1
     images/000002.jpg 1
     images/000003.jpg 1
     …


