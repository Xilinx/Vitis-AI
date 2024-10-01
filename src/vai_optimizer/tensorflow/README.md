<h1 align="center">Prune Model with Vitis Optimizer TensorFlow2</h1>

Vitis Optimizer TensorFlow2 provides a method called ***Iterative pruning*** for model pruning, this method belongs to the coarse-grained pruning category.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Iterative Pruning](#iterative-pruning)
- [Examples](#examples)
- [Results](#results)

## Overview
Vitis Optimizer TensorFlow2 only supports Keras models created by the [Functional API](https://www.tensorflow.org/guide/keras/functional/) or the [Sequential API](https://www.tensorflow.org/guide/keras/sequential_model). [Subclassed](https://www.tensorflow.org/guide/keras/custom_layers_and_models) models are not supported.

## Installation
 We provide two ways to install: [From Source](#from-source) and [Docker Image](#docker-image), respectively.

### From Source
It is recommended to use [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) environment and TensorFlow >= 2.4.0.

Create a conda enviroment as an example:
```shell
$ conda create -n tf_optimizer python=3.8
```

Activate the conda envirment and install Tensorflow 2.12 as an example:
```shell
$ source activate tf_optimizer
$ pip install -r requirements.txt
$ pip install tensorflow==2.12.0
```

Export the environment variable `CUDA_HOME`:
```shell
$ export CUDA_HOME=/usr/local/cuda
```
Or, if you want to install with ROCM support: 

```shell
$ export ROCM_HOME=/opt/rocm
$ export CPPFLAGS=$(hipconfig --cpp_config)
```

**Install the tensorflow optimizer**
If you're not going to modify/debug code yourself, use:
```shell
$ python setup.py install 
```
Or, if you change the code frequently, install with the following command allows you to conveniently edit code.
```shell
$ python setup.py develop 
```
Use the following command to validate that the installation is done.
```shell
$ python -c "from tf_nndct import IterativePruningRunner" 
```
### Docker Image
Vitis AI provides a Docker environment for the Vitis AI Optimizer. The Docker image encapsulates the required tools and libraries necessary for pruning in these frameworks. To get and run the Docker image, please refer to https://xilinx.github.io/Vitis-AI/3.5/html/docs/install/install.html#leverage-vitis-ai-containers.

To get the GPU acceleration support, prebuilt ROCm docker image can be got by:
```shell
$ docker pull xilinx/vitis-ai-tensorflow2-rocm:latest
```
For CUDA docker image, there is no prebuilt one and you have to build it yourself.
You can read [this](https://xilinx.github.io/Vitis-AI/3.5/html/docs/install/install.html#option-2-build-the-docker-container-from-xilinx-recipes) for detailed instructions.

## Iterative Pruning

In short, this method includes two stages: model analysis and pruned model generation. The elaborate prune process is as follows.
### Creating a Baseline Model

Here, a simple MNIST convnet from the [Keras vision example](https://keras.io/examples/vision/mnist_convnet) is used.

```python
model = keras.Sequential([
    keras.Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(), layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"),
])
```
### Creating a Pruning Runner

To create an input specification with shape and dtype and to use this specification to get a pruning runner, use the following command:

```python
from tf_nndct import IterativePruningRunner

input_shape = [28, 28, 1]
input_spec = tf.TensorSpec((1, *input_shape), tf.float32)
runner = IterativePruningRunner(model, input_spec)
```

### Pruning the Baseline Model
To prune a model, follow these steps:

1. Define a function to evaluate model performance. The function must satisfy two requirements:
    1. The first argument must be a keras.Model instance to be evaluated.
    2. Returns a Python number to indicate the performance of the model.

```python
def evaluate(model):
  model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
  score = model.evaluate(x_test, y_test, verbose=0)
  return score[1]
```
2. Use this evaluation function to run model analysis:

```python
runner.ana(evaluate)
```
3. Determine a pruning ratio. The ratio indicates the reduction in the amount of floating-point computation of the model in forward pass.

    pruned_model's FLOPs = (1 – ratio) * original_model's FLOPs
    The value of ratio should be in (0, 1):

```python
sparse_model = runner.prune(ratio=0.2)
```

**Note:** `ratio` is only an approximate target value and the actual pruning ratio may not be exactly equal to this value.

The returned model from `prune()` is sparse which means the pruned channels are set to zeros and model size remains unchanged.
The sparse model is used in the iterative pruning process.
The sparse model is converted to a pruned dense model only after pruning is completed.

Besides returning a sparse model, the pruning runner generates a specification file in the `.vai` directory that describes how each layer will be pruned.

### Fine-tuning a Sparse Model

Training a sparse model is no different from training a normal model. The model will maintain sparsity internally. There is no need for any additional actions other than adjusting the hyper-parameters.

```python
sparse_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
sparse_model.fit(x_train, y_train, batch_size=128, epochs=15, validation_split=0.1)
sparse_model.save_weights("model_sparse_0.2", save_format="tf")
```
**Note:** When calling `save_weights`, use the "tf" format to save the weights.

### Performing Iterative Pruning

Load the checkpoint saved from the previous fine-tuning stage to the model. Increase the ratio value to get a sparser model.
Then continue to fine-tune this sparse model. Repeat this pruning and fine-tuning loop a couple of times until the sparsity reaches the desired value.

```python
model.load_weights("model_sparse_0.2")

input_shape = [28, 28, 1]
input_spec = tf.TensorSpec((1, *input_shape), tf.float32)
runner = IterativePruningRunner(model, input_spec)
sparse_model = runner.prune(ratio=0.5)
```
### Getting the Pruned Model

When the iterative pruning is completed, a sparse model is generated which has the same number of parameters as the original model but with many of them now set to zero.

Call `get_slim_model()` to remove zeroed parameters from the sparse model and retrieve the pruned model:

```python
model.load_weights("model_sparse_0.5")

input_shape = [28, 28, 1]
input_spec = tf.TensorSpec((1, *input_shape), tf.float32)
runner = IterativePruningRunner(model, input_spec)
runner.get_slim_model()
```

By default, the runner uses the latest pruning specification to generate the slim model. You can see what the latest specification file is with the following command:

```shell
$ cat .vai/latest_spec
$ ".vai/mnist_ratio_0.5.spec"
```
If this file does not match your sparse model, you can explicitly specify the file path to be used:

```python
runner.get_slim_model(".vai/mnist_ratio_0.5.spec")
```

## Examples
Please find examples [here](/example/pruning/tensorflow2)

## Results
Here we show some of the prune experiment results. We conducted the ResNet-50, VGG-16, and MobileNetV1 pruning experiments with the ImageNet dataset. All the experiments are run on AMD MI100/MI210 GPUs. For user, with different parameter settings may have different classification precision results.

**ResNet50**

| model                        | FLOPs(G) | Accuray Top1/Top5   |
| ---------------------------- | -------- | ----------- |
| Resnet50  baseline           | 7.73     | 75.13/92.20 |
| prune ratio = 0.2 | 6.16    | 75.80/92.67 |
| prune ratio = 0.3 | 5.39    | 75.29/92.41 |
| prune ratio = 0.4 | 4.62    | 75.11/92.40 |
| prune ratio = 0.5 | 3.84    | 74.74/92.31 |


**VGG-16**

| model                      | FLOPs(G) | Accuray Top1/Top5   |
| -------------------------- | -------- | ----------- |
| VGG-16 baseline            | 30.96    | 70.93/89.82 |
| prune ratio = 0.2 | 24.97    | 71.24/90.00 |
| prune ratio = 0.4 | 18.38    | 69.93/89.53 |
| prune ratio = 0.6 | 12.46    | 67.45/87.76 |

**MobileNetV1**

| model                           | FLOPs(G) | Accuray Top1/Top5   |
| ------------------------------- | -------- | ----------- |
| MobileNetV1  baseline           | 1.15     | 70.75/89.53 |
| prune ratio = 0.1 | 1.03     | 70.31/89.17 |
| prune ratio = 0.2 | 0.92     | 70.07/89.00 |
| prune ratio = 0.3 | 0.80     | 69.62/88.73 |
| prune ratio = 0.4 | 0.70     | 68.71/88.28 |
