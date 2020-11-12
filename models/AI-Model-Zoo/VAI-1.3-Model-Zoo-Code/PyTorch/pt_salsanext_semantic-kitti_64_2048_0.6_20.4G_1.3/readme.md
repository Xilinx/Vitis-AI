# Salsanext Pruned
### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Eval](#eval)
4. [Performance](#performance)
5. [Model_info](#model_info)

### Installation
1. Environment requirement
    - pytorch, opencv, ...
    - vai_q_pytorch(Optional, required by quantization)
    - XIR Python frontend (Optional, required for dumping xmodel)

2. Installation with Docker

   a. Please refer to [vitis-ai](https://github.com/Xilinx/Vitis-AI/tree/master/) for how to obtain the docker image.

   b. Activate pytorch virtual envrionment in docker:
   ```shell
   conda activate vitis-ai-pytorch
   ```
   c. Install all the python dependencies using pip:
   ```shell
   pip install -r requirements.txt
   ```

### Preparation

1. dataset describle.
  ```
  dataset includes image file, groundtruth file and a validation image list file.
  ```
2. prepare datset.

  ```shell

  # please download Semantic_Kitti dataset (http://semantic-kitti.org/dataset.html).
  # put the grundtruth folder and image folder in  `data` directory.

  * `data` directory structure like:
    + data
        +dataset
            + sequences
                + 00
                    + velodyne
                      ▪ 000000.bin
                      ▪ 000001.bin
                    + voxels
                      ▪ 000000.bin
                      ▪ 000000.label
                      ▪ 000000.invalid
                      ▪ 000000.occluded
                      ▪ 000001.bin
                      ▪ 000001.label
                      ▪ 000001.invalid
                      ▪ 000001.occluded

  ```

### Eval

1. Evaluation
  ```shell
  cd ./code/
  # modify configure if you need, includes data root, weight path,...
  bash run_eval.sh
  ```
2. Training 
  ```shell
  cd ./code/
  # modify configure if you need, includes data root, weight path,...
  bash run_train.sh
  ```
3. Quantize and quantized model evaluation
  ```shell
  cd ./code/
  # modify configure if you need, includes data root, weight path,...
  bash run_quant.sh
  ```
  
### Performance

| Model | Input | FLOPs | Performance on Validation| 
|---- |----|----|----------------------------------|
| Salsanext_pruned|1x5x64x2048|20.4G|mIou = 51%|

| Model | INT8 Performance on Validation| 
|-------|----------------------------------|
| Salsanext_pruned|mIou = 45.4%|

### Model_info

1. data preprocess
```
1.1). projection
1.2). mean and scale
    img_means: #range,x,y,z,signal
      - 12.12
      - 10.88
      - 0.23
      - -1.04
      - 0.21
    img_stds: #range,x,y,z,signal
      - 12.32
      - 11.47
      - 6.91
      - 0.86
      - 0.16
```
