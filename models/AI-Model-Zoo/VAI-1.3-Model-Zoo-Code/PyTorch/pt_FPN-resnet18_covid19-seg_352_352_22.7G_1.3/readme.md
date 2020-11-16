# COVID-19 Multi-classes Joint Segmentation using FPN_ResNet18
### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Eval/Test](#eval/test)
4. [Performance](#performance)
5. [Model_info](#model_info)

### Installation
1. Environment requirement
    - pytorch, opencv, tqdm ...
    - vai_q_pytorch(Optional, required by quantization)
    - XIR Python frontend (Optional, required by quantization)

2. Installation with Docker

   - Please refer to [vitis-ai](https://github.com/Xilinx/Vitis-AI/tree/master/) for how to obtain the docker image.

   - Activate pytorch virtual envrionment in docker:
   ```shell
   conda activate vitis-ai-pytorch
   ```
   - Install all the python dependencies using pip:
   ```shell
   pip install -r requirements.txt
   ```

### Preparation

1. Dataset description
   - download dataset (refer this repo https://github.com/lindawangg/COVID-Net)

2. Dataset diretory structure
   ```
   + data/COVID19-seg
     + TrainingSet
       + LungInfection-Train
       + MultiClassInfection-Train
     + TestingSet
       + LungInfection-Test
       + MultiClassInfection-Test
     + val.txt
    ```

### Eval/Test

1. Test & Evaluation
    ```shell
    bash run_eval.sh
    ```
2. Demo
    ```shell
    bash run_demo.sh
    ```
3. Quantization
    ```shell
    bash run_quant.sh
    ```
    
### Performance
Note：the performance is evaluated on COVID19-seg/TestingSet

| model | input size | FLOPs | 2-classes Dice (%)| 3-classes mIoU (%) |
|-------|------------|--------------|---------------|-------|
| FPN-R18 (light-weight)| 352x352 | 22.7G | 85.88% | 59.89% |

| model | input size | FLOPs | INT8 2-classes Dice (%)| INT8 3-classes mIoU (%) |
|-------|------------|--------------|---------------|-------|
| FPN-R18 (light-weight)| 352x352 | 22.7G | 85.47% | 59.57% |

### Model_info

1. Data preprocess
  ```
  data channel order: BGR(0~255)                  
  input size: h * w = 352 * 352
  mean = (104.00698793, 116.66876762, 122.67891434)
  input = (input - mean) 
  ``` 
