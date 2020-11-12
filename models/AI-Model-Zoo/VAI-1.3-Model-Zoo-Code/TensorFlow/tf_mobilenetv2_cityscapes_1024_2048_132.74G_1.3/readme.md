### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Performance](#performance)
5. [Model_info](#model_info)

### Installation

1. Environment requirement
   tensorflow 1.15

2. Installation
   - Create virtual envrionment and activate it:
   ```shell
   conda create -n tf_seg python=3.6
   conda activate tf_seg
   ```
   - Install all the python dependencies using pip:
   ```shell
   pip install -r requirements.txt
   ```

### Preparation

1. Dataset description

    - check dataset soft link or download cityscapes dataset (https://www.cityscapes-dataset.com/downloads)
    - grundtruth folder: gtFine_trainvaltest.zip [241MB]
    - image folder: leftImg8bit_trainvaltest.zip [11GB]

2. Dataset diretory structure
   ```
   + data
     + cityscapes
       + leftImg8bit
         + train
         + val
       + gtFine
         + train
         + val
    ```

### Train/Eval

1. Evaluation
    ```shell
    bash code/test/run_eval.sh
    ```
2. Visulization
    ```shell
    # select one image from validation set for visulization
    cp data/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png  ./code/demo/frankfurt/
    bash code/test/run_demo.sh
    ```
### Performance

| model | input size | Val. mIoU (%)| 
|-------|------------|--------------|
| Deeplabv3+(MobileNetv2)| 1024x2048 | 62.63 | 

### Model_info

1. Data preprocess
  ```
  data channel order: BGR(0~255)                  
  mean = 127.5
  input = input - mean
  ``` 
