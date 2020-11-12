# Classification using Unified Vitis-AI Python APIs

This example uses unified Vitis-AI Python APIs to run `inception_v1` on alveo platforms. 

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
python inception_v1.py <number of threads> <compiler output directory>
```

**Example**

```Python
cd /workspace/alveo/examples/vitis_ai_alveo_samples/inception_v1_mt_py
python inception_v1.py 1 ./model
```

## Classification results display on console window

```sh
Top[0] 0.358162 Cardigan, Cardigan Welsh corgi
Top[1] 0.131761 Bernese mountain dog
Top[2] 0.090558 Appenzeller
Top[3] 0.090558 Border collie
Top[4] 0.054926 Pembroke, Pembroke Welsh corgi
```

>**:pushpin: NOTE:** `vitis-ai-caffe` or `vitis-ai-tensorflow` conda environment must be activated to run this example.


