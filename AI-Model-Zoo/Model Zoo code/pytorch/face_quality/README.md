### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [demo](#demo)
4. [Train/Eval](#traineval)
5. [Performance](#performance)
6. [Model_info](#model_info)

### Installation

1. Environment requirement
    - pytorch 1.1

### Preparation

You should ensure that all datasets used have been face detected.

The released code can currently only be used to train and test model on preprocessed dataset.

Preprocess: The original image-->face detection-->crop face-->resize to H_96xW_72-->save processed image

1. Train/Test dataset
   ``` 
   face quality training and testing datasets are private.
   ```

2. Train/Test dataset diretory structure
   ```
   + data
     + face_quality_dataset
        + images
            + 1.jpg
            + 2.jpg
            + ...
        + train_list.txt
        + test_list.txt
   ```
3. train_list.txt and test_list.txt like: 
   ```
   five_points_0.jpg x0 x1 x2 x3 x4 y0 y1 y2 y3 y4
   five_points_1.jpg x0 x1 x2 x3 x4 y0 y1 y2 y3 y4 
   ...
   five_points_m.jpg x0 x1 x2 x3 x4 y0 y1 y2 y3 y4
   quality_img_0.jpg label_quality_score
   quality_img_1.jpg label_quality_score
   ...
   quality_img_n.jpg label_quality_score
   ```
   ```
   landmark(5 points):x0 x1 x2 x3 x4 y0 y1 y2 y3 y4
   (five points, left-eye, right-eye, nose, left-mouth-corner, right-mouth-corner)
   label_quality_score: You may get this label by pretrain model.
   ```
 

### demo

   ```shell
   cd code/test
   python demo.py --pretrained $PATH_TO_MODEL
   ```


### Train/Eval

1. Train
    ```shell
    cd code/train
    CUDA_VISIBLE_DEVICES="0" python train.py --pretrained $PATH_TO_PRETRAIN_MODEL
    ```
2. Eval
    ```shell
    cd code/train
    CUDA_VISIBLE_DEVICES="0" python train.py --pretrained $PATH_TO_PRETRAIN_MODEL -e
    ```
3. Quantization
    ```shell
    cd code/quantize
    bash quant.sh
    ```

### Performance(private dataset)

|model|l1-loss|
|-|-|
|float|12.33|


|model|l1-loss|
|-|-|
|INT8|12.75|
### Model_info

1. Data preprocess
  ```
  input: cropped face, no padding, resized to 80x60
  output: 5 points and quality(before normalization to score)
  data channel: gray-3channel                  
  Image hegiht and width: h * w = 80* 60
  mean: 0.5
  std: 0.5
  ``` 
