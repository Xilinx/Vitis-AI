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
# Clone repo
$ git clone gits@xcdl190260:cp/vai_utf.git --recursive
# CPU-only version
$ conda build vai_q_tensorflow2_cpu_feedstock --output-folder ./conda_pkg/
# GPU version
$ conda build vai_q_tensorflow2_gpu_feedstock --output-folder ./conda_pkg/
# Install conda package on your machine
$ conda install --use-local ./conda_pkg/linux-64/*.tar.bz2
```

## Test Environment

* Python 3.6, 3.7
* Tensorflow 2.10.0
* Keras 2.10.0

## Quick Start

* #### Post Training Quantization(PTQ)

```python
from tensorflow_model_optimization.quantization.keras import vitis_quantize
# 1. Create a quantizer with proper quantize_strategy.
# a) For DPU devices, set quantize_strategy to 'pof2s' to apply power-of-2 scale quantization. This is the default quantize_strategy.
# b) For other devices supporting floating point arithmetic, set quantize_strategy to 'fs' to apply float scale quantization.
quantizer = vitis_quantize.VitisQuantizer(float_model, quantize_strategy='pof2s')
# 2. Call quantizer.quantize_model to do post training quantization.
# Here calib_dataset is a representative dataset for calibration, you can use full or subset of eval_dataset or train_dataset.
# See 'quantize_model method' section below for detailed options.
quantized_model = quantizer.quantize_model(calib_dataset=calib_dataset)
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

## Inspecting vai_q_tensorflow2

Vitis Inspector is an experimental new feature introduced in Vitis AI 3.0 to inspect a float model and show partition results for a given DPU target architecture, together with some indications on why the layers are not mapped to DPU. Without target, we can only show some general, target-independent inspection results. Assign target to get more detailed inspect results for it.

Note: This feature in only available for default pof2s quantize strategy due to DPU limitations. This feature is experimental, please contact us if you encounter any bugs or problems.

The following codes show how to inspect a model.

```python
model = tf.keras.models.load_model(‘float_model.h5’)
from tensorflow_model_optimization.quantization.keras import vitis_inspect
inspector = vitis_inspect.VitisInspector(target="DPUCZDX8G_ISA1_B4096")
inspector.inspect_model(model,
                        input_shape=[1, 224, 224, 3],
                        dump_model=True,
                        dump_model_file="inspect_model.h5",
                        plot=True,
                        plot_file="model.svg",
                        dump_results=True,
                        dump_results_file="inspect_results.txt",
                        verbose=0)
```

**Arguments**:
*  **target**: string, the target DPU to deploy this model. It can be a name string (for example, DPUCZDX8G_ISA1_B4096), a JSON file path (for example, ./U50/arch.json) or a fingerprint. The default value is None, if the target DPU is not specified, an error will be reported.
*  **model**: tf.keras.Model instance, the float model to be inspected. Float model should have concrete input shapes. Build the float model with concrete input shapes or call inspect_model with the input_shape argument.
*  **input_shape**: list(int) or list(list(int)) or tuple(int) or dictionary(int),  contains the input shape for each input layer. Use default shape info in the input layers if not set. Use list of shapes for multiple inputs, for example inspect_model(model, input_shape=[1, 224, 224, 3]) or inspect_model(model, input_shape=\[[None, 224, 224, 3], [None, 64]]). All dimensions should have concrete values, and batch_size dimension should be None or int.  If the input shape of the model is variable like [None, None, None, 3], you need to specify a shape like [None, 224, 224, 3] to generate the final quantized model. The default value is None.
*  **dump_model**: bool, whether to dump the inspected model and save model to disk. The default value is False.
*  **dump_model_file**: string, path of inspected model file. The default value is inspect_model.h5.
*  **plot**: bool, whether to plot the model inspect results by graphviz and save image to disk. It is helpful when you need to visualize the model inspection results together with some modification hints. Note that only part of output file types can show the hints, such as .svg.The default value is False.
*  **plot_file**: string, file path of model image file when plotting the model. The default value is model.svg.
*  **dump_results**: bool, whether to dump the inspect results and save text to disk. More detailed layer-by-layer results than screen logging will be dumped to the text file. The default value is True.
*  **dump_results_file**: string, file path of inspect results text file. The default value is inspect_results.txt.
*  **verbose**: int, the logging verbosity level. More detailed logging results will be shown for higher verbose value. The default value is 0.

**Known issue**
* **multi-outputs pattern issue**: 1) Due to Xcompiler's pattern matching problem, when the convolution or add layer has multiple output layers and one of which is a relu activation layer, the result of relu layer may be incorrect. 2) When the relu-like activation layer is followed by multiple convolutional layers, the result of convolutional layer may be incorrect. This issue will be fixed in a later version.

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
You can find a full example in [here](https://github.com/Xilinx/Vitis-AI/blob/master/src/Vitis-AI-Quantizer/vai_q_tensorflow2.x/tensorflow_model_optimization/python/examples/quantization/keras/vitis/mnist_cnn_ptq.py).

```python
float_model = tf.keras.models.load_model(‘float_model.h5’)
from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantizer = vitis_quantize.VitisQuantizer(float_model)
quantized_model = quantizer.quantize_model(calib_dataset=calib_dataset, 
                                           calib_steps=100, 
					   calib_batch_size=10, 
					   input_shape=[None, 224, 224, 3])
```

Arguments:
*  **calib_dataset** is used as a representative calibration dataset for calibration as an example.
You can use full or part of eval_dataset, train_dataset or other datasets. You can also use train_dataset or other datasets.
*  **calib_steps** is the total number of steps for calibration. Ignored with the default value of None.
If "calib_dataset" is a tf.data dataset, generator or keras.utils.Sequence instance and steps is None, calibration will run until the dataset is exhausted.
This argument is not supported with array inputs.
*  **calib_batch_size** is the number of samples per batch for calibration.
If the "calib_dataset" is in the form of a dataset, generator or keras.utils.Sequence instances, the batch size is controlled by the dataset itself.
If the "calib_dataset" is in the form of a numpy.array object, the default batch size is 32.
*  **input_shape**: list(int) or list(list(int)) or tuple(int) or dictionary(int),  contains the input shape for each input layer. Use default shape info in the input layers if not set. Use list of shapes for multiple inputs, for example input_shape=[1, 224, 224, 3] or input_shape=\[[None, 224, 224, 3], [None, 64, 1]]. All dimensions should have concrete values, and batch_size dimension should be None or int. If the input shape of the model is variable like [None, None, None, 3], you need to specify a shape like [None, 224, 224, 3] to generate the final quantized model.

3.  #### vai_q_tensorflow2 Fast Finetuning

Generally, there is a small accuracy loss after quantization, but for some networks such as MobileNets, the accuracy loss can be large.
Fast finetuning uses the AdaQuant algorithm to adjust the weights and quantize params layer by layer with the unlabeld calibration dataset and can get better accuracy for some models.
It will take longer time than normal PTQ (still shorter than QAT as calib_dataset is much smaller than train dataset) and is disabled by default to save time,
and can be turned on to try to improve the performance if you see accuracy issues. A recommanded workflow is to first try PTQ without fast finetuning and try quantization with fast finetuning
if the accuracy is not acceptable. QAT is another method to improve the accuracy but may takes longer time and need the train dataset. You can do fast finetuning by setting 'include_fast_ft' to True when
doing the post-training quantization.

```python
quantized_model = quantizer.quantize_model(calib_dataset=calib_dataset, calib_steps=None, calib_batch_size=None, include_fast_ft=True, fast_ft_epochs=10)
```

Arguments:
*  **include_fast_ft** indicates wether to do fast finetuning or not.
*  **fast_ft_epochs** indicates the number of finetuning epochs for each layer.

4.  #### Saving the Quantized Model
The quantized model object is a standard tf.keras model object. You can save it by running the following command:

```python
quantized_model.save('quantized_model.h5')
```

The generated quantized_model.h5 file can be fed to the vai_c_tensorflow compiler and then deployed on the DPU.

5.  #### (Optional) Exporting the Quantized Model to ONNX
The following codes show how to perform post-training quantization and export the quantized model to onnx with vai_q_tensorflow2 API.

```python
model = tf.keras.models.load_model(‘float_model.h5’)
from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantizer = vitis_quantize.VitisQuantizer(model)
quantized_model = quantizer.quantize_model(calib_dataset=calib_dataset, 
                                           output_format='onnx',
                                           onnx_opset_version=11,
                                           output_dir='./quantize_results',
                                           **kwargs) 
```

Arguments:
*  **output_format**: A string object, indicates what format to save the quantized model. Options are: '' for skip saving, 'h5' for saving .h5 file, 'tf' for saving saved_model file, 'onnx' for saving .onnx file. Default to ''.
*  **onnx_opset_version**: An int object, the ONNX opset version. Take effect only when output_format is 'onnx'. Default to 11.
*  **output_dir**: A string object, indicates the directory to save the quantized model in. Default to './quantize_results'.

6.  #### (Optional) Evaluating the Quantized Model
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

7.  #### (Optional) Dumping the Simulation Results
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
quantizer = vitis_quantize.VitisQuantizer(model, quantize_strategy='pof2s_tqt')
qat_model = quantizer.get_qat_model(
    init_quant=True, # Do init PTQ quantization will help us to get a better initial state for the quantizers, especially for `pof2s_tqt` strategy. Must be used together with calib_dataset
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

Note: In Vitis 1.4 we add a new strategy `pof2s_tqt` which uses Trained-Threshold in quantizers and may better results for QAT. By default, the Straight-Through-Estimator will be used.
Please note that `pof2s_tqt` strategy should only be used in QAT and be used together with 'init_quant=True' to get best performance.
Do init PTQ quantization will be helpful to get a better initial state for the quantizer parameters, especially for `pof2s_tqt` strategy, otherwise the training may not converge.

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

## Configure the Quantize Strategy

We provide some default quantize strategies, but sometimes users need to modify quantize configurations for different targets or to get better performance. For example, some target devices may need the biases to be quantized into 32 bit and some may need to quantize only part of the model. In this part we will show you how to configure the quantizer to meet your need.

#### 1. Quantize Strategy

Three main configurable parts of the quantize tool are the quantize tool pipeline, what part of the model to be quantized and how to quantize them. We define all these thing in quantize_strategy. Internally, each quantize_strategy is a JSON file containing below configurations:

* **pipeline_config**: These configurations control the work pipeline of the quantize tool, including some optimizations during quantization, e.g. whether to fold Conv2D + BatchNorm layers, whether to perform Cross-Layer-Equalization algorithm and so on. It can be further divided into optimize_pipeline_config, quantize_pipeline_config, refine_pipeline_config and finalize_pipeline_config.

* **quantize_registry_config**: These configurations control what layer types are quantizable, where to insert the quantize ops and what kind of quantize op to be inserted. It includes some layer specific configurations and user-defined global configurations.

Below is an example configuration for the Conv2D layers:

```python
{
  "layer_type": "tensorflow.keras.layers.Conv2D",
  "quantizable_weights": ["kernel"],
  "weight_quantizers": [
  {
    "quantizer_type": "Pof2SQuantizer",
    "quantizer_params": {"bit_width": 8,"method":0, "round_mode": 1, "symmetry": true, "per_channel": true, "channel_axis": -1, "unsigned": false, "narrow_range": false}
  }],
  "quantizable_biases": ["bias"],
  "bias_quantizers": [
  {
    "quantizer_type": "Pof2SQuantizer",
    "quantizer_params": {"bit_width": 8,"method":0, "round_mode": 1, "symmetry": true, "per_channel": false, "channel_axis": -1, "unsigned": false, "narrow_range": false}
  }],
  "quantizable_activations": ["activation"],
  "activation_quantizers": [
  {
    "quantizer_type": "FSQuantizer",
    "quantizer_params": {"bit_width": 8, "method":2, "method_percentile":99.9999, "round_mode": 1, "symmetry": true, "per_channel": false, "channel_axis": -1, "unsigned": false, "narrow_range": false}
  }]
}
```

As you can see, by using this quantize configuration, we quantize the weight, bias and activations of the Conv2D layer. The weight and biasrs, all of them are using `Pof2SQuantizer`(power-of-2 scale quantizer) and the activation are using `FSQuantizer`(float scale quantizer). We can apply different quantizers for different objects in one layer.

Note: The Quantizer here in configurations means the quantize operation applied to each object. It consumes a float tensor and output a quantized tensor. Please note that the quantization is 'fake', which means that the input is quantized to int and then de-quantized to float.

#### 2. Using Built-in Quantize Strategy

Users can use dump_quantize_strategy to see get the JSON file of current quantize strategy. To make things simple, we provide 4 types of built-in quantize strategies for common user cases which users can extend or override for their need, including:
* pof2s: power-of-2 scale quantization, mainly used for DPU targets now. Default quantize strategy of the quantizer.
* pof2s_tqt: power-of-2 scale quantization with trained thresholds, mainly used for QAT in DPU now.
* fs: float scale quantization, mainly used for devices supporting floating-point calculation, such as CPU/GPUs.
* fsx: trained quantization threshold for power-of-2 scale quantization, mainly used for QAT for DPU now.

Users can switch between the built-in quantize strategies by assigning quantize_strategy argument in the contruct function of VitisQuantizer. Moreover, we provide 2 handy ways to configure the quantize strategy.

#### 3. Configure by kwargs in VitisQuantizer.quantize_model()
This is a easy way for users who need to override the default pipeline configurations or do global modifications on the quantize operations. The kwargs here is a dict object which keys match the quantize configurations in the JSON file. See vitis_quantize.VitisQuantizer.quantize_model for more information about available keys.

Example codes below shows how to use it.

```python
model = tf.keras.models.load_model(‘float_model.h5’)
from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantizer = vitis_quantize.VitisQuantizer(model)
quantizer.quantize_model(calib_dataset,
                         input_layers=['conv2'],
                         bias_bit=32,
                         activation_bit=32,
                         weight_per_channel=True)
```

In this example, we configure the quantizer to quantize part of the model. Layers before conv2 will be not be optimized or quantized. Moreover, we quantize all the activations and biases to 32 bit instead of 8 bit, and use per_channel quantization for all weights.

#### 4. Configure by VitisQuantizer.set_quantize_strategy()

For advanced users who want fully control of the quantize tool, we provide this API to set new quantize strategies JSON file. Users can first dump the current configurations to JSON file and make modifications on the it. This allows users to override the default configurations, make more fine-grained quantizer configurations or extend the quantize config to make more layer types quantizable. Then the user can set the new JSON file to the quantizer to apply these modifications.

Example codes below shows how to do it.

```python
quantizer = VitisQuantizer(model)
# Dump the current quantize strategy
quantizer.dump_quantize_strategy(dump_file='my_quantize_strategy.json', verbose=0)

# Make modifications of the dumped file 'my_quantize_strategy.json'
# Then, set the modified json to the quantizer and do quantization
quantizer.set_quantize_strategy(new_quantize_strategy='my_quantize_strategy.json')
quantizer.quantize_model(calib_dataset)
```

Note: verbose is an int type argument which controls the verbosity of the dumped JSON file. Greater verbose value will dump more detailed quantize strategy. Setting verbose to value greater or equal to 2 will dump the full quantize strategy.

## Quantizing with Float Scale

The quantization for DPU uses power-of-2 scales, symmetry, per-tensor quantizers and need some special processes to simulate DPU behaviors. For other devices supporting floating point scales will need a different quantize strategy, so we introduced the float scale quantization in this release.

* **The fs quantize strategy**: Do quantization for inputs and weights of Conv2D, DepthwiseConv2D, Conv2DTranspose and Dense layers. By default, it will not do Conv-BN folding.

* **The fsx quantize strategy**: Do quantization for more layer types than fs quantize strategy, such as Add, MaxPooling2D and AveragePooling2D. Moreover, it also quantizes the biases and activations of Conv2D, DepthwiseConv2D, Conv2DTranspose and Dense layers. By default, it will do Conv-BN folding.

Note: fs and fsx strategies are designed for target devices with floating-point supports. DPU does not have floating-point support now, so models quantized with these quantize strategies can not be deployed to them.

Users can switch to use float scale quantization by setting quantize_strategy to fs or fsx in the construct function of VitisQuantizer, example codes are showed as below:

```python
model = tf.keras.models.load_model(‘float_model.h5’)
from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantizer = vitis_quantize.VitisQuantizer(model, quantize_strategy='fs')
quantized_model = quantizer.quantize_model(calib_dataset=calib_dataset,
                                           calib_steps=100,
                                           calib_batch_size=10,
                                           **kwargs)
```

* **calib_dataset**: calib_dataset is used as a representative calibration dataset for calibration. You can use full or part of the eval_dataset, train_dataset, or other datasets.
* **calib_steps**: calib_steps is the total number of steps for calibration. It has a default value of None. If calib_dataset is a tf.data dataset, generator, or keras.utils.Sequence instance and steps is None, calibration will run until the dataset is exhausted. This argument is not supported with array inputs.
* **calib_batch_size**: calib_batch_size is the number of samples per batch for calibration. If the "calib_dataset" is in the form of a dataset, generator, or keras.utils.Sequence instances, the batch size is controlled by the dataset itself. If the calib_dataset is in the form of a numpy.array object, the default batch size is 32.
* **kwargs**: dict of the user-defined configurations of quantize strategy. It will override the default built-in quantize strategy. For example, setting bias_bit=16 will let the tool to quantize all the biases with 16bit quantizers. See vai_q_tensorflow2 Usage section for more information of the user-defined configurations.

## Converting to Float16 or BFloat16
vai_q_tensorflow2 supports data type conversions for float models, including Float16, BFloat16, Float, and Double. The following codes show how to perform the data type conversions with vai_q_tensorflow2 API.

```python
model = tf.keras.models.load_model(‘float_model.h5’)
from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantizer = vitis_quantize.VitisQuantizer(model)
quantized_model = quantizer.quantize_model(convert_datatype='float16'
                                           **kwargs)
```

*  **convert_datatype**: A string object, indicates the target data type for the float model. Options are 'float16', 'bfloat16', 'float32', and 'float64'. Default value is 'float16'.

## Quantizing with custom layers

Tensorflow 2 provides a lot of common built-in layers to build the machine learning models, as well as easy ways to for you to write your own
application-specific layers either from scratch or as the composition of existing layers. `Layer` is one of the central abstractions in tf.keras,
subclassing Layer is the recommended way to create custom layers. Please refer to tensorflow [user guide](https://www.tensorflow.org/guide/keras/custom_layers_and_models) for more information.

Vai_q_tensorflow2 provides support for new custom layers via subclassing, including quantizing models with custom layers and an experimental
support for quantizing the custom layers with custom quantize strategies.

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

Here float model contains a custom layer named "MyCustomLayer". The `custom_objects` argument in tf.keras.model.load_model API is needed to load it.
Similarly, VitisQuantizer class provide 'custom_objects' argument to handle the custom layers, below is an example and you can find a full example in [example](https://github.com/Xilinx/Vitis-AI/blob/master/tools/Vitis-AI-Quantizer/vai_q_tensorflow2.x/tensorflow_model_optimization/python/examples/quantization/keras/vitis/mnist_cnn_ptq_custom_layer.py).
The argument `custom_objects` is a dict containing the `{"custom_layer_class_name":"custom_layer_class"}`, multiple custom layers should be separated by a comma.
Moreover, `add_shape_info` should also be set to True for the `quantize_model` API when quantizing models with custom layers to add shape inference information for them.

```python
from tensorflow_model_optimization.quantization.keras import vitis_quantize
# Register the custom layer to VitisQuantizer by custom_objects argument.
quantizer = vitis_quantize.VitisQuantizer(float_model, custom_objects={'MyCustomLayer': MyCustomLayer})
quantized_model = quantizer.quantize_model(calib_dataset=calib_dataset, calib_steps=100, calib_batch_size=10, add_shape_info=True)
```

During the quantization, these custom layers will be wrapped by CustomLayerWrapper and kept unquantized.

Notes:
1. When calling the `dump_model` API to dump golden results for data checking during deployment, we need to set `dump_float=True` to dump float weights and activation for
the custom layers, since they are not quantized.

## (Experimental)Quantizing custom layers with custom quantize strategy

With the default quantize strategy, the custom layers will not be quantized and remain float during quantization, as they are not in the list of supported APIs of vai_q_tensorflow2.
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
quantized_model = quantizer.quantize_model(calib_dataset=calib_dataset, calib_steps=100, calib_batch_size=10)
```

## vai_q_tensorflow2 Supported Operations and APIs

The following table lists the supported operations and APIs for vai_q_tensorflow2.

Table 4. vai_q_tensorflow2 Supported Layers

|Layer Types| Supported Layers | Description |
| ----- | ------ | ------ |
| Core | tf.keras.layers.InputLayer ||
| Core | tf.keras.layers.Dense ||
| Core | tf.keras.layers.Activation | If 'activation' is 'relu' or 'linear', will be quantized.<br>If 'activation' is 'sigmoid' or 'swish', will be converted to hard-sigmoid or hard-swish and then be quantized by default.<br>Otherwise will not be quantized.|
| Convolution | tf.keras.layers.Conv2D ||
| Convolution | tf.keras.layers.DepthwiseConv2D ||
| Convolution | tf.keras.layers.SeparableConv2D ||
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
| Activation | tf.keras.layers.LeakyReLU | Only 'alpha'=0.1 is supported to run on DPU (0.1 will be converted to 26/256 by the quantizer), otherwise will not be quantized and mapped to CPU. |
| Hard_sigmoid | tf.keras.layers.ReLU(6.)(x+3.)*(1./6.) | The supported hard_sigmoid is from [Mobilenet_v3](https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/python/keras/applications/mobilenet_v3.py#L440). <br>`tf.keras.Activation.hard_sigmoid` is not supported now and will not be quantized. |
| Activation | tf.keras.layers.PReLU | The quantization of PReLU is not supported to run on DPU currently, it will be quantized and mapped to CPU. |

## vai_q_tensorflow2 APIs

#### vitis_inspect.VitisInspector Class

```python
vitis_inspect.VitisInspector(
    target=None)
```

The construction function of class VitisInspector.

**Arguments**
* **target**: string or None, the target DPU to deploy this model. It can be a name string (for example, DPUCZDX8G_ISA1_B4096), a JSON file path (for example, ./U50/arch.json) or a fingerprint. The default value is None, if the target DPU is not specified, an error will be reported.

#### inspect_model Method

```python
VitisInspector.inspect_model(model,
                             input_shape=None,
                             dump_model=True,
                             dump_model_file="inspect_model.h5",
                             plot=True,
                             plot_file="model.svg",
                             dump_results=True,
                             dump_results_file="inspect_results.txt",
                             verbose=0)
```

This function performs float model inspection.

**Arguments**
*  **model**: tf.keras.Model instance, the float model to be inspected. Float model should have concrete input shapes. Build the float model with concrete input shapes or call inspect_model with the input_shape argument.
*  **input_shape**: list(int) or list(list(int)) or tuple(int) or dictionary(int),  contains the input shape for each input layer. Use default shape info in the input layers if not set. Use list of shapes for multiple inputs, for example inspect_model(model, input_shape=[1, 224, 224, 3]) or inspect_model(model, input_shape=\[[None, 224, 224, 3], [None, 64]]). All dimensions should have concrete values, and batch_size dimension should be None or int.  If the input shape of the model is variable like [None, None, None, 3], you need to specify a shape like [None, 224, 224, 3] to generate the final quantized model. The default value is None.
*  **dump_model**: bool, whether to dump the inspected model and save model to disk. The default value is False.
*  **dump_model_file**: string, path of inspected model file. The default value is 'inspect_model.h5'.
*  **plot**: bool, whether to plot the model inspect results by graphviz and save image to disk. It is helpful when you need to visualize the model inspection results together with some modification hints. Note that only part of output file types can show the hints, such as .svg.The default value is False.
*  **plot_file**: string, file path of model image file when plotting the model. The default value is model.svg.
*  **dump_results**: bool, whether to dump the inspect results and save text to disk. More detailed layer-by-layer results than screen logging will be dumped to the text file. The default value is True.
*  **dump_results_file**: string, file path of inspect results text file. The default value is inspect_results.txt.
*  **verbose**: int, the logging verbosity level. More detailed logging results will be shown for higher verbose value. The default value is 0.

#### vitis_quantize.VitisQuantizer Class

```python
vitis_quantize.VitisQuantizer(
    model,
    quantize_strategy='pof2s',
    custom_quantize_strategy=None,
    custom_objects={})
```
The construction function of class VitisQuantizer.

**Arguments**

*  **model**: A tf.keras.Model object, containing the configurations for quantization.
* **quantize_strategy**: A string object of the quantize strategy type. Available values are pof2s , pof2s_tqt, fs and fsx. pof2s is the default strategy that uses power-of-2 scale quantizer and the Straight-Through-Estimator. pof2s_tqt is a strategy introduced in Vitis AI 1.4 which uses Trained-Threshold in power-of-2 scale quantizers and may generate better results for QAT. fs is a new quantize strategy introduced in Vitis AI 2.5, it do float scale quantization for inputs and weights of Conv2D, DepthwiseConv2D, Conv2DTranspose and Dense layers. fsx quantize strategy do quantization for more layer types than fs quantize straetgy, such as Add, MaxPooling2D and AveragePooling2D. Moreover, it also quantizes the biases and activations.
*  **custom_quantize_strategy**: A string object, file path of custom quantize strategy json file.
*  **custom_objects**: A Dict object, mapping names(strings) to custom classes or functions.

Note:
1. pof2s_tqt strategy should only be used in QAT and be used together with init_quant=True to get the best performance.
2. fs and fsx strategy are designed for target devices with floating-point supports. DPU does not have floating-point support now, so models quantized with these quantize strategies can not be deployed to them.

#### quantize_model Method

```python
vitis_quantize.VitisQuantizer.quantize_model(
    calib_dataset=None,
    calib_batch_size=None,
    calib_steps=None,
    verbose=0,
    add_shape_info=False,
    **kwargs)
```

This function to do post-training quantization(PTQ) of the float model, including model optimization, weights quantization and activation quantize calibration.

**Arguments**

* **calib_dataset**: A tf.data.Dataset, keras.utils.Sequence, or np.numpy object, the representative dataset for calibration. You can use full or part of eval_dataset, train_dataset, or other datasets as calib_dataset.
* **calib_steps**: An int object, the total number of steps for calibration. Ignored with the default value of None. If "calib_dataset" is a tf.data dataset, generator, or keras.utils.Sequence instance and steps is None, calibration will run until the dataset is exhausted. This argument is not supported with array inputs.
* **calib_batch_size**: An int object, the number of samples per batch for calibration. If the "calib_dataset" is in the form of a dataset, generator, or keras.utils.Sequence instances, the batch size is controlled by the dataset itself. If the "calib_dataset" is in the form of a numpy.array object, the default batch size is 32.
* **verbose**: An int object, the verbosity of the logging. Greater verbose value will generate more detailed logging. Default to 0.
* **add_shape_info**: An bool object, whether to add shape inference information for custom layers. Must be set True for models with custom layers.
* **kwargs**: A dict object, the user-defined configurations of quantize strategy. It will override the default built-in quantize strategy. Detailed user-defined configurations are listed below.

#### Arguments in **kwargs
**kwargs in this API is a dict of the user-defined configurations of quantize strategy. It will override the default built-in quantize strategy. For example, setting "bias_bit=16" will let the tool to quantize all the biases with 16bit quantizers. Detailed user-defined configurations are listed below.

* **separate_conv_act**: A bool object, whether to separate activation functions from the Conv2D/DepthwiseConv2D/TransposeConv2D/Dense layers. Default to True.
* **fold_conv_bn**: A bool object, whether to fold the batch norm layers into previous Conv2D/DepthwiseConv2D/TransposeConv2D/Dense layers. Default to True.
* **convert_bn_to_dwconv**: Named fold_bn in Vitis-AI 2.0 and previous versions. A bool object, whether to convert the standalone BatchNormalization layer into DepthwiseConv2D layers. Default to True.
* **convert_sigmoid_to_hard_sigmoid**: Named replace_sigmoid in Vitis-AI 2.0 previous versions. A bool object, whether to replace the Activation(activation='sigmoid') and Sigmoid layers into hard sigmoid layers and do quantization. If not, the sigmoid layers will be left unquantized and will be scheduled on CPU. Default to True.
* **convert_relu6_to_relu**: Named replace_relu6 in Vitis-AI 2.0 and previous versions. A bool object, whether to replace the ReLU6 layers with ReLU layers. Usually Cross-Layer Equalization algorithm can not be applied to ReLU6 layers, setting this option to True will make it available for those models. However, this conversion may lead to accuracy drop to the float model. Default to False.
* **include_cle**: A bool object, whether to do Cross-Layer Equalization before quantization. Default to True.
* **cle_steps**: A int object, the iteration steps to do Cross-Layer Equalization. Default to 10.
* **cle_to_relu6**: Named forced_cle in Vitis-AI 2.0 and previous versions. A bool object, whether to do forced Cross-Layer Equalization for ReLU6 layers. Default to False.
* **include_fast_ft**: A bool object, whether to do fast fine-tuning or not. Fast fine-tuning adjust the weights layer by layer with calibration dataset and may get better accuracy for some models. Fast fine-tuning is disabled by default. It takes longer than normal PTQ (still much shorter than QAT as calib_dataset is much smaller than the training dataset). Turn on to improve the performance if you meet accuracy issues. Default to False.
* **fast_ft_epochs**: An int object, the iteration epochs to do fast fine-tuning for each layer. Default to 10.
* **output_format**: A string object, indicates what format to save the quantized model. Options are: '' for skip saving, 'h5' for saving .h5 file, 'tf' for saving saved_model file, 'onnx' for saving .onnx file. Default to ''.
* **onnx_opset_version**: An int object, the ONNX opset version. Take effect only when output_format is 'onnx'. Default to 11.
* **output_dir**: A string object, indicates the directory to save the quantized model in. Default to './quantize_results'.
* **convert_datatype**: A string object, indicates the target data type for the float model. Options are 'float16', 'bfloat16', 'float32', and 'float64'. Default value is 'float16'.
* **input_layers**: A list(string) object, names of the start layers to be quantized. Layers before these layers in the model will not be optimized or quantized. For example, this argument can be used to skip some pre-processing layers or stop quantizing the first layer. Default to [].
* **output_layers**: A list(string) object, names of the end layers to be quantized. Layers after these layers in the model will not be optimized or quantized. For example, this argument can be used to skip some post-processing layers or stop quantizing the last layer. Default to [].
* **ignore_layers**: A List(string) object, names of the layers to be ignored during quantization. For example, this argument can be used to skip quantizing some sensitive layers to improve accuracy. Default to [].
* **input_bit**: An int object, the bit width of all inputs. Default to 8.
* **input_method**: An int object, the method to calculate scale factors in quantization of all inputs. Options are: 0 for Non_Overflow, 1 for Min_MSE, 2 for Min_KL, 3 for Percentile. All methods are available for fs and fsx quatize strategies while only 0 and 1 methods are available for pof2s and pof2s_tqt quantize strategies now. Default to 0.
* **input_symmetry**: A bool object, whether to do symmetry or asymmetry quantization for all inputs. Default to True.
* **input_per_channel**: A bool object, whether to do per-channel or per-tensor quantization for all inputs. Default to False.
* **input_round_mode**: An int object, the rounding mode used in quantization of all inputs. Options are: 0 for HALF_TO_EVEN, 1 for HALF_UP, 2 for HALF_AWAY_FROM_ZERO. Default to 1.
* **input_unsigned**: An bool object, whether to use unsigned integer quantization for all inputs. It is usually used for non-negative numeric inputs (such as range from 0 to 1) when input_unsigned is true. Default to False.
* **weight_bit**: An int object, the bit width of all weights. Default to 8.
* **weight_method**: An int object, the method to calculate scale factors in quantization of all weights. Options are: 0 for Non_Overflow, 1 for Min_MSE, 2 for Min_KL, 3 for Percentile. All methods are available for fs and fsx quatize strategies while only 0 and 1 methods are available for pof2s and pof2s_tqt quantize strategies now. Default to 1.
* **weight_symmetry**: A bool object, whether to do symmetry or asymmetry quantization for all weights. Default to True.
* **weight_per_channel**: An bool object, whether to do per-channel or per-tensor quantization for all weights. Default to False.
* **weight_round_mode**: An int object, the rounding mode used in quantization of all weights. Options are: 0 for HALF_TO_EVEN, 1 for HALF_UP, 2 for HALF_AWAY_FROM_ZERO. Default to 0.
* **weight_unsigned**: An bool object, whether to use unsigned integer quantization for all weights. It is usually used when weight_symmetry is false. Default to False.
* **bias_bit**: An int object, the bit width of all biases. Default to 8.
* **bias_method**: An int object, the method to calculate scale factors in quantization of all biases. Options are: 0 for Non_Overflow, 1 for Min_MSE, 2 for Min_KL, 3 for Percentile. All methods are available for fs and fsx quatize strategies while only 0 and 1 methods are available for pof2s and pof2s_tqt quantize strategies now. Default to 0.
* **bias_symmetry**: A bool object, whether to do symmetry or asymmetry quantization for all biases. Default to True.
* **bias_per_channel**: An bool object, whether to do per-channel or per-tensor quantization for all biases. Default to False.
* **bias_round_mode**: An int object, the rounding mode used in quantization of all biases. Options are: 0 for HALF_TO_EVEN, 1 for HALF_UP, 2 for HALF_AWAY_FROM_ZERO. Default to 0.
* **bias_unsigned**: An bool object, whether to use unsigned integer quantization for all bias. It is usually used when bias_symmetry is false. Default to False.
* **activation_bit**: An int object, the bit width of all activations. Default to 8.
* **activation_method**: An int object, the method to calculate scale factors in quantization of all activations. Options are: 0 for Non_Overflow, 1 for Min_MSE, 2 for Min_KL, 3 for Percentile. All methods are available for fs and fsx quatize strategies while only 0 and 1 methods are available for pof2s and pof2s_tqt quantize strategies now. Default to 1.
* **activation_symmetry**: A bool object, whether to do symmetry or asymmetry quantization for all activations. Default to True.
* **activation_per_channel**: An bool object, whether to do per-channel or per-tensor quantization for all activations. Default to False.
* **activation_round_mode**: An int object, the rounding mode used in quantization of all activations. Options are: 0 for HALF_TO_EVEN, 1 for HALF_UP, 2 for HALF_AWAY_FROM_ZERO. Default to 1.
* **activation_unsigned**: An bool object, whether to use unsigned integer quantization for all activations. It is usually used for non-negative numeric activations (such as ReLU or ReLU6) when activation_symmetry is true. Default to False.

#### dump_model Method

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
*  **dump_float**: A bool object, whether to dump the float value of weights and activation results. Default to False.
*  **weights_only**: A bool object, set to True to only dump the weights, set to False will also dump the activation results. Default to False.

#### get_qat_model Method

```python
vitis_quantize.VitisQuantizer.get_qat_model(
    init_quant=False,
    calib_dataset=None,
    calib_batch_size=None,
    calib_steps=None,
    train_with_bn=False,
    freeze_bn_delay=-1)
```

This function to quantize the float model for quantize-aware training(QAT).

**Arguments**

*  **init_quant**: A bool object, whether to do initial quantization before QAT. Do initial PTQ quantization will be helpful to get a better initial state for the quantizer parameters, especially for `pof2s_tqt` strategy, otherwise the training may not converge. Default to False.
*  **calib_dataset**: A tf.data.Dataset, keras.utils.Sequence or np.numpy object, the representative dataset for calibration. Must be set when "init_quant" is set True.
You can use full or part of eval_dataset, train_dataset or other datasets as calib_dataset.
*  **calib_steps**: An int object, the total number of steps for initial quantize calibration. Ignored with the default value of None.
If "calib_dataset" is a tf.data dataset, generator or keras.utils.Sequence instance and steps is None, calibration will run until the dataset is exhausted.
This argument is not supported with array inputs.
*  **calib_batch_size**: An int object, the number of samples per batch for initial quantize calibration.
If the "calib_dataset" is in the form of a dataset, generator or keras.utils.Sequence instances, the batch size is controlled by the dataset itself.
If the "calib_dataset" is in the form of a numpy.array object, the default batch size is 32.
*  **train_with_bn**: A bool object, whether to keep bn layers during quantize-aware training. Default to False.
*  **freeze_bn_delay**: An int object, the train steps before freezing the bn parameters. Default to -1, which means never do bn freezing.

#### get_deploy_model Method

```python
vitis_quantize.VitisQuantizer.get_deploy_model(
    model)
```

This function to convert the QAT models and generates the deployable model, results can be fed into vai_c_tensorflow compiler.

**Arguments**

*  **model**: A tf.keras.Model object, the QAT model to deploy.

## Error Codes
Table 5. vai_q_tensorflow2 error codes

| Error Description | Error Types | Causes and Solutions |
| ----- | ------ | ------ |
| Quantizer_TF2_Unsupported_Layer | Unsupported layer type | Layer is not a `tf.keras.layers.Layer` or this layer is not yet supported. By default, this layer will not be quantized and will be mapped to run on CPU. You can use the experimental support for customizing quantize strategy to define the quantization behaviour of it. |
| Quantizer_TF2_Unsupported_Model | Unsupported model type | Only tf.keras sequential or functional models can be supported. Subclassing model is not supported now, please convert it to sequential or functional model and try again. |
| Quantizer_TF2_Invalid_Input_Shape | Invalid input shape | The input_shape parameter is not valid, please check and set correct value for it. |
| Quantizer_TF2_Invalid_Calib_Dataset | Invalid calibration dataset | The calibration dataset is not valid, please check and set correct value for it. |
| Quantizer_TF2_Invalid_Target | Invalid target | The target parameter is not valid, please check and set correct value for it. |
