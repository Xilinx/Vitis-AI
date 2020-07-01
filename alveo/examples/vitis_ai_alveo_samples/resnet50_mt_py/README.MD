# Classification using Unified Vitis-AI Python APIs

This example uses unified Vitis-AI Python APIs to run `resnet50` on alveo platforms.

## Prerequisite

- Compiler output directory
  - `compiler.json` : File containing low level hardware instructions.
  - `weights.h5` : File containing preprocessed floating point data (weights/biases).
  - `quantizer.json` : File containing scaling factors for each layer in the corresponding network.
  - `meta.json` : File containing library path, xclbin paths.
- Image directory
  - Containing `.jpg` images

### Setup

> **Note:** Skip, If you have already run the below steps.

Activate Conda Environment
  ```sh
  conda activate vitis-ai-caffe (or) conda activate vitis-ai-tensorflow
  ```

Setup the Environment

  ```sh
  source /workspace/alveo/overlaybins/setup.sh
  ```

## Run Inference

The script takes the compiler output directory and an image directory (fixed to `image` in currect path, can be changed) and run classification.

**Syntax**
```Python
python resnet50.py <number of threads> <compiler output directory>
```

**Example**

```Python
cd /workspace/alveo/examples/vitis_ai_alveo_samples/resnet50_mt_py/
python resnet50.py 1 ./model
```

## Classification results display on console window

```sh
Top[0] 0.674899 Cardigan, Cardigan Welsh corgi
Top[1] 0.193362 Pembroke, Pembroke Welsh corgi
Top[2] 0.071134 Border collie
Top[3] 0.020380 collie
Top[4] 0.015872 Shetland sheepdog, Shetland sheep dog, Shetland
```

>**:pushpin: NOTE:** `vitis-ai-caffe` or `vitis-ai-tensorflow` conda environment must be activated to run this example.


