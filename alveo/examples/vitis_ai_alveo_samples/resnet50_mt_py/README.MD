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
Top[0] 0.999950 golf ball
Top[1] 0.000017 tennis ball
Top[2] 0.000013 baseball
Top[3] 0.000010 croquet ball
Top[4] 0.000003 soccer ball
```

>**:pushpin: NOTE:** `vitis-ai-caffe` or `vitis-ai-tensorflow` conda environment must be activated to run this example.


