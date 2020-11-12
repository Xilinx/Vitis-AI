# Semantic Segmentation with ENet_xilinx on Cityscapes
### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
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
    - download cityscapes dataset and unzip the dataset into 'data/cityscapes' (https://www.cityscapes-dataset.com/downloads)
    - groundtruth folder: gtFine_trainvaltest.zip [241MB]
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
    # Select a demo image and put it under ‘data/demo/’ for visualization
    cp data/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png data/demo/
    # Run demo
    bash run_demo.sh
    ```
2. Evaluation
    ```shell
    bash run_eval.sh
    ```
3. Training
    ```shell
    bash run_train.sh
    ```
4. Quantization
    ```shell
    bash run_quant.sh
    ```

### Performance

| Model | Input size | Mean IoU (Validation)| FLOPs |
|-------|------------|--------------|-------|
| ENet_xilinx| 512x1024 | 64.4% | 8.6G |


| model | INT8 Mean IoU (Validation) |
|-------|---------------|
| ENet_xilinx | 63.27% |


### Model_info

1. Data preprocess
  ```
  data channel order: BGR(0~255)                  
  resize: h * w = 512 * 1024 (cv2.resize(image, (w, h)).astype(np.float32))
  mean = (0.485, 0.456, 0.406)
  std =  (0.229, 0.224, 0.225)
  input = input / 255.0
  input = (input - mean) / std
  ``` 

