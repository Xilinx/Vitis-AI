# RefineDet based on pruned Vgg16 for Medical Detection

## Contents

1. [Environment](#Environment)
2. [Preparation](#Preparation)
3. [Eval](#Eval)
4. [Performance](#Performance)
5. [Model information](#Model information)
6. [Quantize](#quantize)

## Environment
1. Environment requirement
   tensorflow 1.15

2. Installation
   - Create virtual envrionment and activate it:
   ```shell
   conda create -n tf_det python=3.6
   conda activate tf_det
   ```
   - Install all the python dependencies using pip:
   ```shell
   pip install -r requirements.txt
   ```
## Preparation

1. Dataset pre-processing.
   Please make soft link of the EDD dataset

   ```
    + data
        + EDD
          + images
          + bbox
          + all_gt.txt
          + val.txt
   ```

## Eval

1. Evaluate the tf pb model
```shell
sh run_eval.sh
```

2. Visulization for the tf pb model.
```shell
cd code/test
python demo.py
```

## Performance
| Input | mAP | FLOPs|
|----|----|---|
|320x320| 78.39%| 9.83G |
  
## Model information
* input image size: 320\*320 
* data channel order: RGB(0~255)    
* name of input node: 'image:0'
* name of output node: 'arm_cls:0', 'arm_loc:0', 'odm_cls:0', 'odm_loc:0'.

### Quantize
1. Quantize tool installation
  See [vai_q_tensorflow](https://github.com/Xilinx/Vitis-AI/tree/master/Vitis-AI-Quantizer/vai_q_tensorflow)

2. Quantize workspace
  See [quantize](./code/quantize/)
