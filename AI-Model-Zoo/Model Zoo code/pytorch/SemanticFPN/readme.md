### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Performance](#performance)
5. [Model_info](#model_info)

### Installation

1. Environment requirement
    - pytorch 
    - opencv, tqdm etc. (Refer to [code/conda_config/environment.yaml](code/conda_config/environment.yaml) for more details.)

2. Quick installation
   ```shell
   conda env create -f code/conda_config/environment.yaml
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

1. Visulization
    ```shell
    bash code/test/run_demo.sh
    ```

2. Evaluation
    ```shell
    bash code/test/run_eval.sh
    ```
3. Training
    ```shell
    bash code/train/run_train.sh
    ```
4. Quantization
    ```shell
    bash code/quantize/run_quant.sh
    ```
### Performance

| model | input size | Val. mIoU (%)| FLops |
|-------|------------|--------------|-------|
| SemanticFPN(ResNet18)| 256x512 | 62.9 | 10G |
| SemanticFPN(MobileNetv2)| 256x512 | 62.6 | 1.35G |
| SemanticFPN(MobileNetv2)| 512x1024 | 68.7 | 5.4G |

| model | INT8 mIoU (%)|
|-------|---------------|
| SemanticFPN(ResNet18)| 60.1 |

### Model_info

1. Data preprocess
  ```
  data channel order: BGR(0~255)                  
  resize: h * w = 256 * 512 (cv2.resize(image, (new_w, new_h)).astype(np.float32))
  mean = (0.485, 0.456, 0.406)
  std = (0.229, 0.224, 0.225)
  input = input / 255.0
  input = (input - mean) / std
  ``` 
