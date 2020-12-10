# **VAI_Q_TENSORFLOW2**

The Xilinx Vitis AI Quantizer for Tensorflow 2.x. 

It is customized based on [tensorflow-model-optimization](https://github.com/tensorflow/model-optimization).
The TensorFlow Model Optimization Toolkit is a suite of tools that users, both novice and advanced, can use to optimize machine learning models for deployment and execution.

## User Document

[**Vitis AI User Document**](https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html#documentation)

## Installation

### Build wheel package and install:
```
$ sh build.sh
$ pip install pkgs/*.whl
```

### Build conda package (need anaconda):
```
# CPU-only version
$ conda build vai_q_tensorflow2_cpu_feedstock --output-folder ./conda_pkg/
# GPU version
$ conda build vai_q_tensorflow2_gpu_feedstock --output-folder ./conda_pkg/
# Install conda package
$ conda install --use-local ./conda_pkg/linux-64/*.tar.bz2
```

## Test Environment


* Python 3.6 
* Tensorflow 2.3.1 

## Usage

* ### Quantize model: 

```python
from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantizer = vitis_quantize.VitisQuantizer(model)    # Here `model` is the created or loaded float model
quantized_model = quantizer.quantize_model(calib_dataset=eval_dataset)      # Here `eval_dataset` is the representative dataset for calibration, you can also use train_dataset

# Then you can save the quantized model just like the float model
quantized_model.save('quantized_model.h5')
```

* ### Evaluate the quantized model 

```python
quantized_model.compile(loss=your_loss, metrics=your_metrics)
quantized_model.evaluate(eval_dataset)
```

Note that the quantize_model() function removes the loss and optimizers information from the original model. So model.compile() is needed to evaluate the model.

* ### Load the saved quantized model: 

The quantized model has some custom objects, so you need to load the model with quantize_scope().

```python
from tensorflow_model_optimization.quantization.keras import vitis_quantize
with vitis_quantize.quantize_scope():
    model = keras.models.load_model('./quantized_model.h5')
```

* ### Dump the quantized weights and activation for DPU debug: 

```python
from tensorflow_model_optimization.quantization.keras import vitis_quantize
vitis_quantize.VitisQuantizer.dump_model(quantized_model, dump_dataset)
```

Note that the batch_size of the dump_dataset should be set to 1 for DPU debugging.
