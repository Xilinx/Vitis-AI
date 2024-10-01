# movenet


### Contents
1. [Use Case and Application](#Use-Case-and-Application)
2. [Specification](#Specification)
3. [Paper and Architecture](#Paper-and-Architecture)
4. [Dataset Preparation](#Dataset-Preparation)
5. [Use Guide](#Use-Guide)
6. [License](#License)
7. [Note](#Note)


### Use Case and Application

   - Pose Estimation
   - Trained on COCO2017 dataset
   
   
### Specification

| Metric             | Value                                   |
| :----------------- | :-------------------------------------- |
| Framework          | PyTorch                                 |
| Prune Ratio        | 0%                                      |
| FLOPs              | 0.5G                                    |
| Input Dims         | 192,192,3                               |
| FP32 Accuracy      | 0.7972                                  |
| INT8 Accuracy      | 0.7981                                  |
| Train Dataset      | COCO2017                                |
| Test Dataset       | COCO2017                                |
| Supported Platform | GPU, VEK280, V70                        |
  

### Paper and Architecture 

1. Network Architecture: movenet
 
2. Paper link: https://arxiv.org/abs/2105.04154
  
  
### Dataset Preparation

1.Download COCO dataset2017 from https://cocodataset.org/. (You need train2017.zip, val2017.zip and annotations.)Unzip to `$MOVENET_PATH/code/data/` like this:

```
+-- data
    +-- annotations (person_keypoints_train2017.json, person_keypoints_val2017.json, ...)
    +-- train2017   (xx.jpg, xx.jpg,...)
    +-- val2017     (xx.jpg, xx.jpg,...)

```


2.Make data to our data format.
```
python scripts/make_coco_data_17keypooints.py
```
```
Our data format: JSON file
Keypoints order:['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 
    'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 
    'right_ankle']

One item:
[{"img_name": "0.jpg",
  "keypoints": [x0,y0,z0,x1,y1,z1,...],
  #z: 0 for no label, 1 for labeled but invisible, 2 for labeled and visible
  "center": [x,y],
  "bbox":[x0,y0,x1,y1],
  "other_centers": [[x0,y0],[x1,y1],...],
  "other_keypoints": [[[x0,y0],[x1,y1],...],[[x0,y0],[x1,y1],...],...], #lenth = num_keypoints
 },
 ...
]
```

3.You can add your own data to the same format.


### Use Guide

1. Test and Evaluation
    ```shell

    bash run_test.sh 
    ```
2. Training 

    ```shell

    bash run_train.sh 
    ```
3. Demo
    ```shell

    bash run_demo.sh
    ```
4. Quantization
    ```shell

    bash run_quantize.sh
    ```

### License

Apache License 2.0

For details, please refer to **[Vitis-AI License](https://github.com/Xilinx/Vitis-AI/blob/master/LICENSE)**


### Note

Data preprocess
  ```
  data channel order: RGB(0~255)                  
  input size: h * w = 192 * 192
  ```
  