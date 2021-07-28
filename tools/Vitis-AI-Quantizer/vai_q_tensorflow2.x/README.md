# **VAI_Q_TENSORFLOW2**

The Xilinx Vitis AI Quantizer for Tensorflow 2.x. 

It is customized based on [tensorflow-model-optimization](https://github.com/tensorflow/model-optimization).
The TensorFlow Model Optimization Toolkit is a suite of tools that users, both novice and advanced, can use to optimize machine learning models for deployment and execution.

## User Document

See [**Vitis AI User Document**](https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html#documentation)

## Installation

You can install vai_q_tensorflow2 in the following three ways:

#### Install Using Docker Container
[**Vitis AI**](https://github.com/Xilinx/Vitis-AI) provides a Docker container for quantization tools, including vai_q_tensorflow. After running a container, activate the conda environment vitis-ai-tensorflow2.
```
$ conda activate vitis-ai-tensorflow2
```

(Optional) To install vitis-ai-tensorflow2 patch package inside docker container
```
$ sudo env CONDA_PREFIX=/opt/vitis_ai/conda/envs/vitis-ai-tensorflow2/ PATH=/opt/vitis_ai/conda/bin:$PATH conda install patch_package.tar.bz2
```

#### Install from Source Code with Wheel Package
vai_q_tensorflow2 is a fork of TensorFlow Model Optimization Toolkit. It is open source in Vitis_AI_Quantizer. To build vai_q_tensorflow2, run the following command:
```
$ sh build.sh
$ pip install pkgs/*.whl
```

#### Install from Source Code with Conda Package (need anaconda):
```
# CPU-only version
$ conda build vai_q_tensorflow2_cpu_feedstock --output-folder ./conda_pkg/
# GPU version
$ conda build vai_q_tensorflow2_gpu_feedstock --output-folder ./conda_pkg/
# Install conda package on your machine
$ conda install --use-local ./conda_pkg/linux-64/*.tar.bz2
```

## Test Environment

* Python 3.6, 3.7 
* Tensorflow 2.3.1 

## Quick Start

* #### Post Training Quantization(PTQ)

```python
from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantizer = vitis_quantize.VitisQuantizer(model)    # Here `model` is the created or loaded float model
quantized_model = quantizer.quantize_model(calib_dataset=calib_dataset)      # Here calib_dataset is a representative dataset for calibration, you can use full or subset of eval_dataset or train_dataset
```

* #### Quantize-Aware Training(QAT)

```python
from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantizer = vitis_quantize.VitisQuantizer(model)
qat_model = quantizer.get_qat_model()

# Then run the training process with this qat_model to get the quantize finetuned model.
qat_model.fit(train_dataset)
```

* #### Save the quantized model

```python
# You can save the quantized model just like the float model
quantized_model.save('quantized_model.h5')
```

* #### Load the saved quantized model: 

```python
model = keras.models.load_model('./quantized_model.h5')
```

* #### Evaluate the quantized model 

```python
quantized_model.compile(loss=your_loss, metrics=your_metrics)
quantized_model.evaluate(eval_dataset)
```

* #### Dump the quantized weights and activation for DPU debug: 

```python
vitis_quantize.VitisQuantizer.dump_model(quantized_model, dataset=dump_dataset)
```

Note that the batch_size of the dump_dataset should be set to 1 for DPU debugging.

## Running vai_q_tensorflow2

Vai_q_tensorflow supports two different approaches to quantizing a deep learning model. 
Post-training quantization(PTQ) is a technique to convert a pretrained float model into a quantized model with little degradation in model accuracy.
A representative dataset is needed to run a few batches of inference on the float model to abtain the distributions of the activations, which is also called quantize calibration.
Another approach is quantize-aware training(QAT), which models the quantization errors in both the forward and backward passes during model quantization.
Starting from a float-point pretrained model with good accuracy when doing QAT usually outperforms starting from scratch.

vai_q_tensorflow2 supports both PTQ and QAT and the usage are as follows.

### vai_q_tensorflow2 Post-Training Quantization(PTQ)

Use the following steps to run PTQ with vai_q_tensorflow2.

1.  #### Preparing the Float Model and Calibration Set

Before running vai_q_tensorflow2, prepare the float model and calibration set, including the files listed in the following table.

Table 1. Input Files for vai_q_tensorflow2

| No. | Name | Description |
| ------ | ------ | ----- |
| 1 | float model | Floating-point TensorFlow 2 models, either in h5 format or saved model format. |
| 2 | calibration dataset | A subset of the training dataset or validation dataset to represent the input data distribution, usually 100 to 1000 images are enough. |

2.  #### Quantizing Using the vai_q_tensorflow2 API

Below codes shows how to do post-training quantization with vai_q_tensorflow2 API. 
You can find a full example in [example](https://github.com/Xilinx/Vitis-AI/blob/master/tools/Vitis-AI-Quantizer/vai_q_tensorflow2.x/tensorflow_model_optimization/python/examples/quantization/keras/vitis/mnist_cnn_ptq.py).

```python
float_model = tf.keras.models.load_model(‘float_model.h5’)
from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantizer = vitis_quantize.VitisQuantizer(float_model)
quantized_model = quantizer.quantize_model(calib_dataset=calib_dataset, calib_step=100, calib_batch_size=10)   
```

Notes:
*  **calib_dataset** is used as a representative calibration dataset for calibration as an example. 
You can use full or part of eval_dataset, train_dataset or other datasets. You can also use train_dataset or other datasets. 
*  **calib_steps** is the total number of steps for calibration. Ignored with the default value of None. 
If "calib_dataset" is a tf.data dataset, generator or keras.utils.Sequence instance and steps is None, calibration will run until the dataset is exhausted. 
This argument is not supported with array inputs.
*  **calib_batch_size** is the number of samples per batch for calibration.
If the "calib_dataset" is in the form of a dataset, generator or keras.utils.Sequence instances, the batch size is controlled by the dataset itself.
If the "calib_dataset" is in the form of a numpy.array object, the default batch size is 32.

3.  #### vai_q_tensorflow2 Fast Finetuning

Generally, there is a small accuracy loss after quantization, but for some networks such as MobileNets, the accuracy loss can be large. 
Fast finetuning uses the AdaQuant algorithm to adjust the weights and quantize params layer by layer with the unlabeld calibration dataset and can get better accuracy for some models.
It will take longer time than normal PTQ (still shorter than QAT as calib_dataset is much smaller than train dataset) and is disabled by default to save time, 
and can be turned on to try to improve the performance if you see accuracy issues. A recommanded workflow is to first try PTQ without fast finetuning and try quantization with fast finetuning
if the accuracy is not acceptable. QAT is another method to improve the accuracy but may takes longer time and need the train dataset. You can do fast finetuning by setting 'include_fast_ft' to True when
doing the post-training quantization.

```python
quantized_model = quantizer.quantize_model(calib_dataset=calib_dataset, calib_step=None, calib_batch_size=None, include_fast_ft=True, fast_ft_epochs=10)   
```

Notes:
*  **include_fast_ft** indicates wether to do fast finetuning or not. 
*  **fast_ft_epochs** indicates the number of finetuning epochs for each layer.

4.  #### Saving the Quantized Model
The quantized model object is a standard tf.keras model object. You can save it by running the following command:

```python
quantized_model.save('quantized_model.h5')
```

The generated quantized_model.h5 file can be fed to the vai_c_tensorflow compiler and then deployed on the DPU.

5.  #### (Optional) Evaluating the Quantized Model
If you have scripts to evaluate float models, like the models in Xilinx Model Zoo, you can replace the float model file with the quantized model for evaluation. 
To support the customized quantize layers, the vitis_quantize module should be imported, for example:

```python
from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantized_model = tf.keras.models.load_model('quantized_model.h5')
```

After that, evaluate the quantized model just as the float model, for example:

```python
quantized_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
	metrics= keras.metrics.SparseTopKCategoricalAccuracy())
quantized_model.evaluate(eval_dataset)
```

6.  #### (Optional) Dumping the Simulation Results
Sometimes after deploying the quantized model, it is necessary to compare the simulation results on the CPU/GPU and the output values on the DPU. 
You can use the VitisQuantizer.dump_model API of vai_q_tensorflow2 to dump the simulation results with the quantized_model.

```python
from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantized_model = keras.models.load_model('./quantized_model.h5')
vitis_quantize.VitisQuantizer.dump_model(quantized_model, dump_dataset, output_dir='./dump_results')
```

Note: The batch_size of the dump_dataset should be set to 1 for DPU debugging.

Dump results are generated in ${dump_output_dir} after the command has successfully executed. 
Results for weights and activation of each layer are saved separately in the folder. 
For each quantized layer, results are saved in *.bin and *.txt formats. 
If the output of the layer is not quantized (such as for the softmax layer), the float activation results are saved in *_float.bin and *_float.txt if "save_float" is set True.
The / symbol is replaced by _ for simplicity. Examples for dumping results are shown in the following table.

Table 2. Example of Dumping Results

| Quantized | Layer Name | Saved Weights | Saved Activations|
| ------ | ------ | ----- | ----- |
| Yes | resnet_v1_50/conv1 | {output_dir}/dump_results_weights/quant_resnet_v1_50_conv1_kernel.bin<br>{output_dir}/dump_results_weights/quant_resnet_v1_50_conv1_kernel.txt<br>{output_dir}/dump_results_weights/quant_resnet_v1_50_conv1_bias.bin<br>{output_dir}/dump_results_weights/quant_resnet_v1_50_conv1_bias.txt | {output_dir}/dump_results_0/quant_resnet_v1_50_conv1.bin<br>{output_dir}/dump_results_0/quant_resnet_v1_50_conv1.txt |
| No | resnet_v1_50/softmax | N/A	N/A	| {output_dir}/dump_results_0/quant_resnet_v1_50_softmax_float.bin<br>{output_dir}/dump_results_0/quant_resnet_v1_50_softmax_float.txt |

### vai_q_tensorflow2 Quantize-aware Training(QAT)

Generally, there is a acceptable accuracy degradation after quantization, but for some networks the accuracy loss can be large, such as MobileNets. In this situation, QAT can be used to further improve the accuracy of quantized models.

Technically, quantize finetuning is similar to float model training/finetuning. The difference is that vai_q_tensorflow2 will rewrite the float graph to convert it to a quantized model before the training starts. The typical workflow is as follows.
You can find a full example in [example](https://github.com/Xilinx/Vitis-AI/blob/master/tools/Vitis-AI-Quantizer/vai_q_tensorflow2.x/tensorflow_model_optimization/python/examples/quantization/keras/vitis/mnist_cnn_qat.py).

1.  #### Preparing the Float Model, Dataset and Training Scripts
Before finetuning, please prepare the following files:

Table 3. Input Files for vai_q_tensorflow2 Quantize Finetuning

| No. | Name | Description |
| ------ | ------ | ----- |
| 1 | Float model | Floating-point model files to start finetuning from. Can be omitted if training from scratch. |
| 2 | Dataset | The training and evaluation dataset with labels. |
| 3 | Training Scripts | The Python scripts to run float train/finetuning of the model. |

2.  #### (Optional) Evaluate the Float Model
It is suggested to evaluate the float model first before doing quantize finetuning, which can check the correctness of the scripts and dataset. 
The accuracy and loss values of the float checkpoint can also be a baseline for the quantize finetuning.

3.  #### Modify the Training Scripts And Run Training/Finetuning
Use the vai_q_tensorflow2 API, VitisQuantizer.get_qat_model, to convert the model to quantized model and then do training/finetuning with it. The following is an example:

```python
model = tf.keras.models.load_model(‘float_model.h5’)

# *Call Vai_q_tensorflow2 api to create the quantize training model
from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantizer = vitis_quantize.VitisQuantizer(model, quantize_strategy='8bit_tqt')
qat_model = quantizer.get_qat_model(
    init_quant=True, # Do init PTQ quantization will help us to get a better initial state for the quantizers, especially for `8bit_tqt` strategy. Must be used together with calib_dataset
    calib_dataset=calib_dataset)
    
# Then run the training process with this qat_model to get the quantize finetuned model.
# Compile the model
model.compile(
	optimizer= RMSprop(learning_rate=lr_schedule),		
	loss=tf.keras.losses.SparseCategoricalCrossentropy(),
	metrics=keras.metrics.SparseTopKCategoricalAccuracy())

# Start the training/finetuning
model.fit(train_dataset)
```

Note: In Vitis 1.4 we add a new strategy `8bit_tqt` which uses Trained-Threshold in quantizers and may better results for QAT. By default, the Straight-Through-Estimator will be used.
Please note that `8bit_tqt` strategy should only be used in QAT and be used together with 'init_quant=True' to get best performance. 
Do init PTQ quantization will be helpful to get a better initial state for the quantizer parameters, especially for `8bit_tqt` strategy, otherwise the training may not converge.

4.  #### Save the Model
Call model.save() to save the trained model or use callbacks in model.fit() to save the model periodically. For example:

```python
# save model manually
model.save(‘qat_model.h5’)

# save the model periodically during fit using callbacks
model.fit(
	train_dataset, 
	callbacks = [
      		keras.callbacks.ModelCheckpoint(
          	filepath=’./quantize_train/’
          	save_best_only=True,
          	monitor="sparse_categorical_accuracy",
          	verbose=1,
      )])
```

5.  #### Convert to deployable quantized model
The trained/finetuned model need to be checked and slightly modified to meet the compiler requirements. 
For example, if "train_with_bn" is set True, the bn layers are not really folded during training and need to be folded before deployment, and so is the Dropout layers.
Moreover, some of the quantizer params may vary during training and ran out of the compiler permitted ranges and need to be corrected before deployment.

We provide a get_deploy_model() function to do these conversion and generate the deployable model, the following is an example:

```python
quantized_model = vitis_quantizer.get_deploy_model(model)
quantized_model.save('quantized_model.h5')
```

6.  #### (Optional) Evaluate the Quantized Model
Call model.evaluate() on the eval_dataset to evaluate the quantized model, just like evaluation of the float model.

```python
from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantized_model = tf.keras.models.load_model('quantized_model.h5')

quantized_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
	metrics= keras.metrics.SparseTopKCategoricalAccuracy())
quantized_model.evaluate(eval_dataset)
```

Note: Quantize finetuning works like float finetuning, so it will be of great help to have some experience on float model training and finetuning. 
For example, how to choose hyper-parameters like optimizer type and learning rate.

### Quantizing with custom layers

Some models may have custom layers inside, vai_q_tensorflow2 provides interfaces to load them, for example:

```python
class MyCustomLayer(keras.layers.Layer):

  def __init__(self, units=32, **kwargs):
    super(MyLayer, self).__init__(kwargs)
    self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(
        shape=(input_shape[-1], self.units),
        initializer="random_normal",
        trainable=True,
        name='w')
    self.b = self.add_weight(
        shape=(self.units,), initializer="zeros", trainable=True, name='b')

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b

  def get_config(self):
    base_config = super(MyLayer, self).get_config()
    config = {"units": self.units}
    return dict(list(base_config.items()) + list(config.items()))

# Here is a float model with custom layer "MyCustomLayer", use custom_objects argument in tf.keras.models.load_model to load it.
float_model = tf.keras.models.load_model(‘float_model.h5’, custom_objects={'MyCustomLayer': MyCustomLayer})
```

Here float model contains a custom layer named "MyCustomLayer". To correctly load it into memory, we need to use costom_objects argument in tf.keras.model.load_model API.
Similarly, VitisQuantizer class provide 'custom_objects' argument to handle the custom layers, below is an example and you can find a full example in [example](https://github.com/Xilinx/Vitis-AI/blob/master/tools/Vitis-AI-Quantizer/vai_q_tensorflow2.x/tensorflow_model_optimization/python/examples/quantization/keras/vitis/mnist_cnn_ptq_custom_layer.py).

```python
from tensorflow_model_optimization.quantization.keras import vitis_quantize
# Register the custom layer to VitisQuantizer by custom_objects argument.
quantizer = vitis_quantize.VitisQuantizer(float_model, custom_objects={'MyCustomLayer': MyCustomLayer})
quantized_model = quantizer.quantize_model(calib_dataset=calib_dataset, calib_step=100, calib_batch_size=10)
```

With the default quantize strategy, the custom layers will not bequantized and remain float during quantization, as they are not in the list of supported APIs of vai_q_tensorflow2. 
We provide an interface named 'custom_quantize_strategy' for advanced users to make custom quantize strategies to do quantize experiments on them. The custom quantize strategy is
a Dict object containing the quantize strategy items or a json file of the Dict. 

The [default quantize strategy](https://github.com/Xilinx/Vitis-AI/blob/master/tools/Vitis-AI-Quantizer/vai_q_tensorflow2.x/tensorflow_model_optimization/python/core/quantization/keras/vitis/eight_bit/vitis_8bit_default_quantize_strategy.json) 
provides an example of the quantize strategy, the custom quantize strategy should follow the same format and act in override behaviour, which means the same item in the custom quantize strategy
will override the one in default strategy, but new items will be added to the quantize strategy.

With this feature, we can quantize the 'MuCustomLayer' layer in previous example:

```python
# Define quantizer with custom quantize strategy, which quantizes w,b and outputs 0 of MyCustomLayer objects.
my_quantize_strategy = {
    "quantize_registry_config": {
        "layer_quantize_config": [{
            "layer_type": "__main__.MyCustomLayer",
            "quantizable_weights": ["w", "b"],
            "weight_quantizers": [
                "quantizer_type": "LastValueQuantPosQuantizer","quantizer_params": {"bit_width": 8, "method": 1, "round_mode": 0},     
                "quantizer_type": "LastValueQuantPosQuantizer", "quantizer_params": {"bit_width": 8, "method": 1, "round_mode": 0}
            ],
            "quantizable_outputs": ["0"],
            "output_quantizers": [
                "quantizer_type": "LastValueQuantPosQuantizer", "quantizer_params": {"bit_width": 8, "method": 1, "round_mode": 1}
            ]
        }]
    }
}
quantizer = vitis_quantize.VitisQuantizer(model, custom_objects={'MyLayer': MyLayer}, custom_quantize_strategy=my_quantize_strategy)

# The following quantization process are all the same as before, here we do normal PTQ as an example
quantized_model = quantizer.quantize_model(calib_dataset=calib_dataset, calib_step=100, calib_batch_size=10)
```

## vai_q_tensorflow2 Supported Operations and APIs

The following table lists the supported operations and APIs for vai_q_tensorflow2.

Table 4. vai_q_tensorflow2 Supported Layers

|Layer Types| Supported Layers | Description |
| ----- | ------ | ------ |
| Core | tf.keras.layers.InputLayer ||
| Core | tf.keras.layers.Dense ||
| Core | tf.keras.layers.Activation | If 'activation' is 'relu' or 'linear', will be quantized.<br>If 'activation' is 'sigmoid' or 'swish', will be converted to hard-sigmoid or hard-swish and then be quantized.<br>Otherwise will not be quantized.|
| Convolution | tf.keras.layers.Conv2D ||
| Convolution | tf.keras.layers.DepthwiseConv2D ||
| Convolution | tf.keras.layers.Conv2DTranspose ||
| Pooling | tf.keras.layers.AveragePooling2D ||
| Pooling | tf.keras.layers.MaxPooling2D ||
| Pooling | tf.keras.layers.GlobalAveragePooling ||
| Normalization | tf.keras.layers.BatchNormalization | By default, BatchNormalization layers will be fused with previous convolution layers or converted to depthwise convolutions if not fuseable.<br>In QAT mode, BatchNormalization layers will be fake fused if `train_with_bn` is set True and then be really fused in `get_deploy_model` function. |
| Regularization | tf.keras.layers.Dropout | By default, Dropout layers will be removed. In QAT mode, Dropout layers will be kept if `remove_dropout` is set False and then be really removed in `get_deploy_model` function. |
| Reshaping | tf.keras.layers.Reshape ||
| Reshaping | tf.keras.layers.Flatten ||
| Reshaping | tf.keras.UpSampling2D ||
| Reshaping | tf.keras.ZeroPadding2D ||
| Merging | tf.keras.layers.Concatenate ||
| Merging | tf.keras.layers.Add ||
| Merging | tf.keras.layers.Muliply ||
| Activation | tf.keras.layers.ReLU ||
| Activation | tf.keras.layers.Softmax | Softmax layer's input will be quantized, can run on standalone Softmax IP to accelerate. |
| Activation | tf.keras.layers.LeakyReLU | Only 'alpha'=0.1 is supported to run on DPU, otherwise will not be quantized and mapped to CPU. |

## vai_q_tensorflow2 APIs

#### VitisQuantizer Class

```python
vitis_quantize.VitisQuantizer(
    float_model, 
    quantize_strategy='8bit', 
    custom_quantize_strategy=None, 
    custom_objects={})
```
The construction function of class VitisQuantizer.

**Arguments**

*  **model**: A tf.keras.Model object, containing the configurations for quantization.
*  **quantize_strategy**: A string object, quantize strategy type . Availablehoices are '8bit' and '8bit_tqt'. 
'8bit' is the default strategy which uses the Straight-Through-Estimator.
'8bit_tqt' is a new strategy introduced in Vitis 1.4 which uses Trained-Threshold in quantizers and may better results for QAT. Please note that `8bit_tqt` strategy should only be used in QAT and be used together with 'init_quant=True' to get best performance. 
*  **custom_quantize_strategy**: A string object, file path of custom quantize strategy json file.
*  **custom_objects**: A Dict object, mapping names(strings) to custom classes or functions.

#### quantize_model method

```python
vitis_quantize.VitisQuantizer.quantize_model(
    calib_dataset=None,
    calib_batch_size=None,
    calib_steps=None,
    verbose=0,
    fold_conv_bn=True,
    fold_bn=True,
    replace_relu6=True,
    include_cle=True,
    cle_steps=10,
    forced_cle=False,
    include_fast_ft=False,
    fast_ft_epochs=10)
```

This function to do post-training quantization(PTQ) of the float model, including model optimization, weights quantization and activation quantize calibration.

**Arguments**

*  **calib_dataset**: A tf.data.Dataset, keras.utils.Sequence or np.numpy object, the representative dataset for calibration.
You can use full or part of eval_dataset, train_dataset or other datasets as calib_dataset.
*  **calib_steps**: An int object, the total number of steps for calibration. Ignored with the default value of None. 
If "calib_dataset" is a tf.data dataset, generator or keras.utils.Sequence instance and steps is None, calibration will run until the dataset is exhausted. 
This argument is not supported with array inputs.
*  **calib_batch_size**: An int object, the number of samples per batch for calibration.
If the "calib_dataset" is in the form of a dataset, generator or keras.utils.Sequence instances, the batch size is controlled by the dataset itself.
If the "calib_dataset" is in the form of a numpy.array object, the default batch size is 32.
*  **fold_conv_bn**: A bool object, whether to fold the batchnorm layers into previous Conv2D/DepthwiseConv2D/TransposeConv2D/Dense layers.
*  **fold_bn**: A bool object, whether to convert the standalone batchnorm layer into DepthwiseConv2D layers.
*  **replace_relu6**: A bool object, whether to replace the Relu6 layers with Relu layers.
*  **include_cle**: A bool object, whether to do Cross Layer Equalization before quantization.
*  **cle_steps**: An int object, the iteration steps to do Cross Layer Equalization.
*  **forced_cle**: A bool object, whether to do forced cle for relu6 layers.
*  **include_fast_ft**: A bool object, wether to do fast finetuning or not. Fast finetuning adjust the weights layer by layer with calibration dataset and may get better accuracy for some models.
It will take much longer time than normal PTQ (still shorter than QAT as calib_dataset is much smaller than train dataset) and is disabled by default to save time, 
and can be turned on to try to improve the performance if you see accuracy issues.
*  **fast_ft_epochs**: An int object, the iteration epochs to do fast finetuning for each layer.

#### dump_model method

```python
vitis_quantize.VitisQuantizer.dump_model(
    model,
    dataset=None,
    output_dir=’./dump_results’,
    dump_float=False,
    weights_only=False)
```

This function dumps the simulation results of the quantized model, including weights and activation results.

**Arguments**

*  **model**: A tf.keras.Model object, the quantized model to dump.
*  **dataset**: A tf.data.Dataset, keras.utils.Sequence or np.numpy object, the dataset used to dump, not needed if weights_only is set to True.
*  **output_dir**: A string object, the directory to save the dump results.
*  **dump_float**: A bool object, whether to dump the float value of weights and activation results.
*  **weights_only**: A bool object, set to True to only dump the weights, set to False will also dump the activation results.

#### get_qat_model method

```python
vitis_quantize.VitisQuantizer.get_qat_model(
    init_quant=False,
    calib_dataset=None,
    calib_batch_size=None,
    calib_steps=None,
    train_with_bn=False,
    freeze_bn_delay=-1,
    replace_relu6=True,
    include_cle=True,
    cle_steps=10,
    forced_cle=False)
```

This function to quantize the float model for quantize-aware training(QAT).

**Arguments**

*  **init_quant**: A bool object, whether to do initial quantization before QAT. Do initial PTQ quantization will be helpful to get a better initial state for the quantizer parameters, especially for `8bit_tqt` strategy, otherwise the training may not converge.
*  **calib_dataset**: A tf.data.Dataset, keras.utils.Sequence or np.numpy object, the representative dataset for calibration. Must be set when "init_quant" is set True.
You can use full or part of eval_dataset, train_dataset or other datasets as calib_dataset.
*  **calib_steps**: An int object, the total number of steps for initial quantize calibration. Ignored with the default value of None. 
If "calib_dataset" is a tf.data dataset, generator or keras.utils.Sequence instance and steps is None, calibration will run until the dataset is exhausted. 
This argument is not supported with array inputs.
*  **calib_batch_size**: An int object, the number of samples per batch for initial quantize calibration.
If the "calib_dataset" is in the form of a dataset, generator or keras.utils.Sequence instances, the batch size is controlled by the dataset itself.
If the "calib_dataset" is in the form of a numpy.array object, the default batch size is 32.
*  **train_with_bn**: A bool object, whether to keep bn layers during quantize-aware training.
*  **freeze_bn_delay**: An int object, the train steps before freezing the bn parameters. Default to -1, which means never do bn freezing.
*  **replace_relu6**: A bool object, whether to replace the Relu6 layers with Relu layers.
*  **include_cle**: A bool object, whether to do Cross Layer Equalization before quantization.
*  **cle_steps**: An int object, the iteration steps to do Cross Layer Equalization.
*  **forced_cle**: A bool object, whether to do forced cle for relu6 layers.

#### get_deploy_model method

```python
vitis_quantize.VitisQuantizer.get_deploy_model(model)
```

This function to convert the QAT models and generates the deployable model, results can be fed into vai_c_tensorflow compiler.

**Arguments**

*  **model**: A tf.keras.Model object, the QAT model to deploy.
