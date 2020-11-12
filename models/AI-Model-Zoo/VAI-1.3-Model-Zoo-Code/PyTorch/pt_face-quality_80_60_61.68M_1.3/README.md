### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [demo](#demo)
4. [Train/Eval](#traineval)
5. [Performance](#performance)
6. [Model_info](#model_info)

### Installation

1. Environment requirement
    - pytorch
    - vai_q_pytorch(Optional, required by quantization)
    - XIR Python frontend(Optional, required by quantization)

2. Installation with Docker

   - Please refer to [vitis-ai](https://github.com/Xilinx/Vitis-AI/tree/master/) for how to obtain the docker image.

   - Activate pytorch virtual envrionment in docker:
   ```shell
   conda activate vitis-ai-pytorch
   ```

### Preparation

You should ensure that all datasets used have been face detected.

The released code can currently only be used to train and test model on preprocessed dataset.

Preprocess: The original image-->face detection-->crop face-->resize to H_96xW_72-->save processed image

1. Train/Test dataset
   ``` 
   face quality training and testing datasets are private. So you need prepare your own face dataset.
   ```

2. Train/Test dataset diretory structure
   ```
   + data
     + face_quality
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

1. prepare test images 
   Preprocess: The original image-->face detection-->crop face

2. run demo

   ```shell
   cd code/test
   python demo.py --pretrained $PATH_TO_MODEL
   # Ouput will be found in code/test/results/{image name}_{normalized score}_{the original score}.jpg
   # Reference threshold: 0.4
   # if normalized score > 0.4, the face quality is OK. 
   ```

### Train/Eval

1. Train
    ```shell
    cd code/train
    export W_QUANT=0
    CUDA_VISIBLE_DEVICES="0" python train.py --quant_mode float --pretrained $PATH_TO_PRETRAIN_MODEL
    ```
2. Eval
    ```shell
    cd code/train
    export W_QUANT=0
    CUDA_VISIBLE_DEVICES="0" python train.py --quant_mode float --pretrained $PATH_TO_PRETRAIN_MODEL -e
    ```
3. Quantization
    ```shell
    cd code/train
    export W_QUANT=1
    CUDA_VISUABLE_DEVICES="0" python train.py --quant_mode calib --pretrained $PATH_TO_PRETRAIN_MODEL -e
    python train.py --quant_mode test --device cpu --pretrained $PATH_TO_PRETRAIN_MODEL -e
    python train.py --dump_xmodel --quant_mode test --device cpu --pretrained $PATH_TO_PRETRAIN_MODEL -e
    ```

### Performance(private dataset)

|model|l1-loss|
|-|-|
|float|12.33|


|model|l1-loss|
|-|-|
|INT8|12.58|
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
