### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Performance](#performance)
5. [Model_info](#model_info)

### Installation

1. Environment requirement
    - anaconda3
    - tensorflow 1.15
    - opencv, tqdm, pillow etc.

2. Installation
   ```shell
   conda create -n tf_yolov3_env python=3.6
   source activate tf_yolov3_env
   pip install -r requirements.txt
   ```

### Preparation

1. Dataset description
    The model is trained on VOC2007_trainval + VOC2012_trainval and tested on VOC2007_test.

2. Download and prepare the dataset
    ```shell
    bash code/test/dataset_tools/prepare_data.sh
    ```

3. Dataset diretory structure
   ```
   + data
     + voc2007_test
       + images
         + 000001.jpg
         + 000002.jpg
         + ...
       + test.txt
       + gt_detection.txt
    ```

### Train/Eval

1. Evaluation
    Configure the model path and data path in [code/test/run_eval.sh](code/test/run_eval.sh)
    ```shell
    bash code/test/run_eval.sh
    ```

### Performance

|Model          |Input size |FLOPs  |Params |train set          |val set   |mAP    |
|-              |-          |-      |-      |-                  |-         |-      |
|yolov3_voc     |416x416    |65.63G |61.68M |voc07+12_trainval  |voc07_test|78.46% |


### Model_info

1. Data preprocess
  ```
  data channel order: RGB(0~255)
  input = input / 255
  resize: keep aspect ratio of the raw image and resize it to make the length of the longer side equal to 416
  padding: pad along the short side with 0.5 to generate the input image with size = 416 x 416
  ``` 
2. Node information

  ```
  input node: 'input_1:0'
  output nodes: 'conv2d_59/BiasAdd:0', 'conv2d_67/BiasAdd:0', 'conv2d_75/BiasAdd:0'
  ```

