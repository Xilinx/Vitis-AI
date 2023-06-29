<h1 align="center">New Data Format with Vitis Optimizer Tensorflow</h1>

Vitis Optimizer Tensorflow provides two methods of new data format:
- Block Floating Point (BFP)
- Microsoft Floating Point (MSFP)

## Table of Contents
- [Installation](#installation)
  - [From Source](#from-source)
  - [Docker Image](#docker-image)
- [Block Floating Point](#block-floating-point)
- [Microsoft Floating Point](#microsoft-floating-point)
- [Supported Operations](#supported-operations)
- [Examples](#examples)
- [Results](#results)

## Installation
### From Source
It is recommended to use an [Anaconda](https://www.anaconda.com/distribution/#download-section) environment.

#### Envrionment Setup
Export the environment variable `CUDA_HOME`:

```
$ export CUDA_HOME=/usr/local/cuda
```

#### Install Tensorflow
Take the CUDA 11.6 version of tensorflow 2.9.3 as an example, you can install Tensorflow by:

```
$ pip install tensorflow==2.9.3 tensorflow-gpu==2.9.3
```
For more details, please see https://www.tensorflow.org/install

#### Install from Source Code with Wheel Package

```
$ sh build.sh
$ pip install pkgs/*.whl
```

#### Verify the Installation:

If the following command line does not an report error, the installation is done.
```
$ python -c "import tensorflow_model_optimization"
```

### Docker Image
Vitis AI provides a Docker environment for the Vitis AI Optimizer. The Docker image encapsulates the required tools and libraries necessary for pruning in these frameworks. To get and run the Docker image, please refer to https://xilinx.github.io/Vitis-AI/3.5/html/docs/install/install.html#leverage-vitis-ai-containers.


For CUDA docker image, there is no prebuilt one and you have to build it yourself.
You can read [this](https://xilinx.github.io/Vitis-AI/3.5/html/docs/install/install.html#option-2-build-the-docker-container-from-xilinx-recipes) for detailed instructions.

## Block Floating Point
Suppose "float_model" is what you want to apply "new data format" to.

### Post training Quantization(PTQ)

```python
from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantizer = vitis_quantize.VitisQuantizer(model, quantize_strategy='bfp')
quantized_model = quantizer.quantize_model(calib_dataset=calib_dataset, data_format='bfp')
```

### Quantize-Aware Training(QAT)

```python
from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantizer = vitis_quantize.VitisQuantizer(model, quantize_strategy='bfp')
qat_model = quantizer.get_qat_model(data_format='bfp')
```

## Microsoft Floating Point
The only difference from BFP is changing 'data_format' parameter from 'bfp' to 'msfp'.

## Supported Operations

Operation             |BFP|MSFP
:--------------------:|:-------:|:------:
|tensorflow.keras.layers.Conv2d|:heavy_check_mark:|:heavy_check_mark:
|tensorflow.keras.layers.Dense|:heavy_check_mark:|:heavy_check_mark:
|tensorflow.keras.layers.Conv2DTranspose|:heavy_check_mark:|:heavy_check_mark:
|tensorflow.keras.layers.DepthwiseConv2D|:heavy_check_mark:|:heavy_check_mark:
|tensorflow.keras.layers.MultiHeadAttention|:heavy_check_mark:|:heavy_check_mark:
|tensorflow.keras.layers.GlobalAveragePooling2D|:heavy_check_mark:|:heavy_check_mark:
|tensorflow.keras.layers.Activation|:heavy_check_mark:|:heavy_check_mark:
|tensorflow.keras.layers.Add|:heavy_check_mark:|:heavy_check_mark:

## Examples
Please find examples [here](tensorflow_model_optimization/python/examples/quantization/keras/vitis/fashion_mnist_bfp)

## Results

<pre>
========================================================================================================
Model                               fp32          bfp13         msfp13        bfp16         msfp16      
========================================================================================================         
mobilenet_v1_tf2_from_tf1           71.02/89.99   26.52/48.69   36.02/60.94   70.15/89.50   70.52/89.74
mobilenet-v2-1.4-224_tf2_from_tf1   74.05/91.87   14.90/30.43   42.85/67.01   72.53/91.09   73.76/91.71
resnet_v1.5_50-tf2_from_tf1         76.52/93.07   72.70/90.99   75.17/92.37   76.42/93.07   76.40/93.06
resnet50v2_tf2_from_tf1             75.59/92.83   73.10/91.65   74.34/92.22   75.50/92.82   75.52/92.81
vgg16_tf2_from_tf1                  70.89/89.85   70.40/89.60   70.63/89.69   70.92/89.83   70.88/89.85
vgg19_tf2_from_tf1                  71.00/89.85   70.43/89.59   70.66/89.69   70.97/89.82   71.00/89.83
inceptionv3_tf2_from_tf1            77.98/93.94   74.74/92.38   75.82/92.85   77.86/93.91   77.86/93.92
inceptionv4_tf2_from_tf1            80.18/95.19   77.27/93.69   78.39/94.19   80.18/95.19   80.19/95.19
mobilenet_v2_tf2_from_tf1           70.13/89.53   6.47/14.94    29.18/52.00   68.57/88.64   69.51/89.20
nasnet_mobile_tf2_from_tf1          73.97/91.58   68.72/88.46   71.17/89.99   73.81/91.56   73.79/91.48
nasnet-large_tf2_from_tf1           82.71/96.17   82.27/95.98   82.40/96.02   82.69/96.16   82.68/96.17
________________________________________________________________________________________________________
</pre>

The results in the table are not final.
