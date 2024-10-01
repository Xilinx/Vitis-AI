# **VAI_Q_ONNX**

The Xilinx Vitis AI Quantizer for ONNX models.

It is customized based on [Quantization Tool](https://github.com/microsoft/onnxruntime/tree/rel-1.14.0/onnxruntime/python/tools/quantization) in ONNX Runtime.


## Test Environment

* Python 3.7, 3.8
* onnx>=1.12.0
* onnxruntime>=1.14.0
* onnxruntime-extensions>=0.4.2

## Installation

You can install vai_q_onnx in the following way:

#### Install from Source Code with Wheel Package
To build vai_q_onnx, run the following command:
```
$ sh build.sh
$ pip install pkgs/*.whl
```


## Quick Start

* #### Post Training Quantization(PTQ) - Static Quantization

The static quantization method first runs the model using a set of inputs called calibration data. During these runs, we compute the quantization parameters for each activation. These quantization parameters are written as constants to the quantized model and used for all inputs. Our quantization tool supports the following calibration methods: MinMax, Entropy and Percentile, MinMSE.

```python
import vai_q_onnx

vai_q_onnx.quantize_static(
    model_input,
    model_output,
    calibration_data_reader,
    quant_format=vai_q_onnx.VitisQuantFormat.FixNeuron,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE)
```

**Arguments**

* **model_input**: (String) This parameter represents the file path of the model to be quantized.
* **model_output**: (String) This parameter represents the file path where the quantized model will be saved.
* **calibration_data_reader**: (Object or None) This parameter is a calibration data reader. It enumerates the calibration data and generates inputs for the original model. If you wish to use random data for a quick test, you can set calibration_data_reader to None. The default value is None.
* **quant_format**: (String) This parameter is used to specify the quantization format of the model. It has the following options:
<br>**QOperator:** This option quantizes the model directly using quantized operators.
<br>**QDQ:** This option quantizes the model by inserting QuantizeLinear/DeQuantizeLinear into the tensor. It supports 8-bit quantization only.
<br>**VitisQuantFormat.QDQ:** This option quantizes the model by inserting VAIQuantizeLinear/VAIDeQuantizeLinear into the tensor. It supports a wider range of bit-widths and configurations.
<br>**VitisQuantFormat.FixNeuron:** This option quantizes the model by inserting FixNeuron (a combination of QuantizeLinear and DeQuantizeLinear) into the tensor.
* **calibrate_method**: (String) For DPU devices, set calibrate_method to either 'vai_q_onnx.PowerOfTwoMethod.NonOverflow' or 'vai_q_onnx.PowerOfTwoMethod.MinMSE' to apply power-of-2 scale quantization. The PowerOfTwoMethod currently supports two methods: MinMSE and NonOverflow. The default method is MinMSE.


## Running vai_q_onnx


Quantization in ONNX Runtime refers to the linear quantization of an ONNX model. We have developed the vai_q_onnx tool as a plugin for ONNX Runtime to support more post-training quantization(PTQ) functions for quantizing a deep learning model. Post-training quantization(PTQ) is a technique to convert a pre-trained float model into a quantized model with little degradation in model accuracy. A representative dataset is needed to run a few batches of inference on the float model to obtain the distributions of the activations, which is also called quantized calibration.


vai_q_onnx supports static quantization and the usage is as follows.

### vai_q_onnx Post-Training Quantization(PTQ)

Use the following steps to run PTQ with vai_q_onnx.

1.  #### Preparing the Float Model and Calibration Set

Before running vai_q_onnx, prepare the float model and calibration set, including the files listed in the following table.

Table 1. Input files for vai_q_onnx

| No. | Name | Description |
| ------ | ------ | ----- |
| 1 | float model | Floating-point ONNX models in onnx format. |
| 2 | calibration dataset | A subset of the training dataset or validation dataset to represent the input data distribution, usually 100 to 1000 images are enough. |

2. #### (Recommended) Pre-processing on the Float Model

Pre-processing is to transform a float model to prepare it for quantization. It consists of the following three optional steps:

* Symbolic shape inference: This is best suited for transformer models.
* Model Optimization: This step uses ONNX Runtime native library to rewrite the computation graph, including merging computation nodes, and eliminating redundancies to improve runtime efficiency.
* ONNX shape inference.

The goal of these steps is to improve quantization quality. ONNX Runtime quantization tool works best when the tensor’s shape is known. Both symbolic shape inference and ONNX shape inference help figure out tensor shapes. Symbolic shape inference works best with transformer-based models, and ONNX shape inference works with other models.

Model optimization performs certain operator fusion that makes the quantization tool’s job easier. For instance, a Convolution operator followed by BatchNormalization can be fused into one during the optimization, which can be quantized very efficiently.

Unfortunately, a known issue in ONNX Runtime is that model optimization can not output a model size greater than 2GB. So for large models, optimization must be skipped.

Pre-processing API is in the Python module onnxruntime.quantization.shape_inference, function quant_pre_process().

```python
from onnxruntime.quantization import shape_inference

shape_inference.quant_pre_process(
     input_model_path: str,
    output_model_path: str,
    skip_optimization: bool = False,
    skip_onnx_shape: bool = False,
    skip_symbolic_shape: bool = False,
    auto_merge: bool = False,
    int_max: int = 2**31 - 1,
    guess_output_rank: bool = False,
    verbose: int = 0,
    save_as_external_data: bool = False,
    all_tensors_to_one_file: bool = False,
    external_data_location: str = "./",
    external_data_size_threshold: int = 1024,)
```

**Arguments**

* **input_model_path**: (String) This parameter specifies the file path of the input model that is to be pre-processed for quantization.
* **output_model_path**: (String) This parameter specifies the file path where the pre-processed model will be saved.
* **skip_optimization**:  (Boolean) This flag indicates whether to skip the model optimization step. If set to True, model optimization will be skipped, which may cause ONNX shape inference failure for some models. The default value is False.
* **skip_onnx_shape**:  (Boolean) This flag indicates whether to skip the ONNX shape inference step. The symbolic shape inference is most effective with transformer-based models. Skipping all shape inferences may reduce the effectiveness of quantization, as a tensor with an unknown shape cannot be quantized. The default value is False.
* **skip_symbolic_shape**:  (Boolean) This flag indicates whether to skip the symbolic shape inference step. Symbolic shape inference is most effective with transformer-based models. Skipping all shape inferences may reduce the effectiveness of quantization, as a tensor with an unknown shape cannot be quantized. The default value is False.
* **auto_merge**: (Boolean) This flag determines whether to automatically merge symbolic dimensions when a conflict occurs during symbolic shape inference. The default value is False.
* **int_max**:  (Integer) This parameter specifies the maximum integer value that is to be considered as boundless for operations like slice during symbolic shape inference. The default value is 2**31 - 1.
* **guess_output_rank**: (Boolean) This flag indicates whether to guess the output rank to be the same as input 0 for unknown operations. The default value is False.
* **verbose**: (Integer) This parameter controls the level of detailed information logged during inference. A value of 0 turns off logging, 1 logs warnings, and 3 logs detailed information. The default value is 0.
* **save_as_external_data**: (Boolean) This flag determines whether to save the ONNX model to external data. The default value is False.
* **all_tensors_to_one_file**: (Boolean) This flag indicates whether to save all the external data to one file. The default value is False.
* **external_data_location**: (String) This parameter specifies the file location where the external file is saved. The default value is "./".
* **external_data_size_threshold**:  (Integer) This parameter specifies the size threshold for external data. The default value is 1024.

3.  #### Quantizing Using the vai_q_onnx API
The static quantization method first runs the model using a set of inputs called calibration data. During these runs, we compute the quantization parameters for each activation. These quantization parameters are written as constants to the quantized model and used for all inputs. Vai_q_onnx quantization tool has expanded calibration methods to power-of-2 scale/float scale quantization methods. Float scale quantization methods include MinMax, Entropy, and Percentile. Power-of-2 scale quantization methods include MinMax and MinMSE.

```python

vai_q_onnx.quantize_static(
    model_input,
    model_output,
    calibration_data_reader,
    quant_format=vai_q_onnx.VitisQuantFormat.FixNeuron,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    input_nodes=[],
    output_nodes=[],
    extra_options=None,)
```


**Arguments**

* **model_input**: (String) This parameter specifies the file path of the model that is to be quantized.
* **model_output**: (String) This parameter specifies the file path where the quantized model will be saved.
* **calibration_data_reader**: (Object or None) This parameter is a calibration data reader that enumerates the calibration data and generates inputs for the original model. If you wish to use random data for a quick test, you can set calibration_data_reader to None.
* **quant_format**: (Enum) This parameter defines the quantization format for the model. It has the following options:
<br>**QOperator** This option quantizes the model directly using quantized operators.
<br>**QDQ** This option quantizes the model by inserting QuantizeLinear/DeQuantizeLinear into the tensor. It supports 8-bit quantization only.
<br>**VitisQuantFormat.QDQ** This option quantizes the model by inserting VAIQuantizeLinear/VAIDeQuantizeLinear into the tensor. It supports a wider range of bit-widths and configurations.
<br>**VitisQuantFormat.FixNeuron** This option quantizes the model by inserting FixNeuron (a combination of QuantizeLinear and DeQuantizeLinear) into the tensor. This is the default value.
* **calibrate_method**:  (Enum) This parameter is used to set the power-of-2 scale quantization method for DPU devices. It currently supports two methods: 'vai_q_onnx.PowerOfTwoMethod.NonOverflow' and 'vai_q_onnx.PowerOfTwoMethod.MinMSE'. The default value is 'vai_q_onnx.PowerOfTwoMethod.MinMSE'.
* **input_nodes**:  (List of Strings) This parameter is a list of the names of the starting nodes to be quantized. Nodes in the model before these nodes will not be quantized. For example, this argument can be used to skip some pre-processing nodes or stop the first node from being quantized. The default value is an empty list ([]).
* **output_nodes**: (List of Strings) This parameter is a list of the names of the end nodes to be quantized. Nodes in the model after these nodes will not be quantized. For example, this argument can be used to skip some post-processing nodes or stop the last node from being quantized. The default value is an empty list ([]).
* **extra_options**: (Dict or None) This parameter is a dictionary of additional options that can be passed to the quantization process. If there are no additional options to provide, this can be set to None. The default value is None.




4.  #### (Optional) Evaluating the Quantized Model
If you have scripts to evaluate float models, like the models in Xilinx Model Zoo, you can replace the float model file with the quantized model for evaluation.

To support the customized FixNeuron op, the vai_dquantize module should be imported, for example:

```python
import onnxruntime as ort
from vai_q_onnx.operators.vai_ops.qdq_ops import vai_dquantize

so = ort.SessionOptions()
so.register_custom_ops_library(_lib_path())
sess = ort.InferenceSession(dump_model, so)
input_name = sess.get_inputs()[0].name
results_outputs = sess.run(None, {input_name: input_data})
```

After that, evaluate the quantized model just as the float model.


5.  #### (Optional) Dumping the Simulation Results
Sometimes after deploying the quantized model, it is necessary to compare the simulation results on the CPU/GPU and the output values on the DPU.
You can use the dump_model API of vai_q_onnx to dump the simulation results with the quantized_model.

```python
# This function dumps the simulation results of the quantized model,
# including weights and activation results.
vai_q_onnx.dump_model(
    model,
    dump_data_reader=None,
    output_dir='./dump_results',
    dump_float=False)
```

**Arguments**

* **model**: (String) This parameter specifies the file path of the quantized model whose simulation results are to be dumped.
* **dump_data_reader**:  (Object or None) This parameter is a data reader that is used for the dumping process. It generates inputs for the original model. 
* **output_dir**: (String) This parameter specifies the directory where the dumped simulation results will be saved. After successful execution of the function, dump results are generated in this specified directory. The default value is './dump_results'.
* **dump_float**: (Boolean) This flag determines whether to dump the floating-point value of weights and activation results. If set to True, the float values will be dumped. The default value is False.

Note: The batch_size of the dump_data_reader should be set to 1 for DPU debugging.

Dump results are generated in output_dir after the command has been successfully executed.
Results for weights and activation of each node are saved separately in the folder.
For each quantized node, results are saved in *.bin and *.txt formats.
If the output of the node is not quantized (such as for the softmax node), the float activation results are saved in *_float.bin and *_float.txt if "save_float" is set to True.
Examples of dumping results are shown in the following table.

Table 2. Example of Dumping Results

| Quantized | Node Name | Saved Weights and Activations|
| ------ | ------ | ----- |
| Yes | resnet_v1_50_conv1 | {output_dir}/dump_results/quant_resnet_v1_50_conv1.bin<br>{output_dir}/dump_results/quant_resnet_v1_50_conv1.txt|
| Yes | resnet_v1_50_conv1_weights | {output_dir}/dump_results/quant_resnet_v1_50_conv1_weights.bin<br>{output_dir}/dump_results/quant_resnet_v1_50_conv1_weights.txt  |
| No | resnet_v1_50_softmax | {output_dir}/dump_results/quant_resnet_v1_50_softmax_float.bin<br>{output_dir}/dump_results/quant_resnet_v1_50_softmax_float.txt |


## List of Vai_q_onnx Supported Quantized Ops

The following table lists the supported operations and APIs for vai_q_onnx.

Table 3. List of Vai_q_onnx Supported Quantized Ops
| supported ops | Comments |
| :-- | :-- |
| Add| |
| Conv| |
| ConvTranspose| |
| Gemm| |
| Concat| |
| Relu| |
| Reshape| |
| Transpose| |
| Resize| |
| MaxPool| |
| GlobalAveragePool| |
| AveragePool| |
| MatMul| |
| Mul| |
| Sigmoid| |
| Softmax| |


## vai_q_onnx APIs

quantize_static Method

```python
vai_q_onnx.quantize_static(
    model_input,
    model_output,
    calibration_data_reader,
    quant_format=vai_q_onnx.VitisQuantFormat.FixNeuron,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    input_nodes=[],
    output_nodes=[],
    op_types_to_quantize=None,
    per_channel=False,
    reduce_range=False,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    nodes_to_quantize=None,
    nodes_to_exclude=None,
    optimize_model=True,
    use_external_data_format=False,
    extra_options=None,)
```


**Arguments**


* **model_input**: (String) This parameter specifies the file path of the model that is to be quantized.
* **model_output**: (String) This parameter specifies the file path where the quantized model will be saved.
* **calibration_data_reader**: (Object or None) This parameter is a calibration data reader that enumerates the calibration data and generates inputs for the original model. If you wish to use random data for a quick test, you can set calibration_data_reader to None.
* **quant_format**: (Enum) This parameter defines the quantization format for the model. It has the following options:
<br>**QOperator** This option quantizes the model directly using quantized operators.
<br>**QDQ** This option quantizes the model by inserting QuantizeLinear/DeQuantizeLinear into the tensor. It supports 8-bit quantization only.
<br>**VitisQuantFormat.QDQ** This option quantizes the model by inserting VAIQuantizeLinear/VAIDeQuantizeLinear into the tensor. It supports a wider range of bit-widths and configurations.
<br>**VitisQuantFormat.FixNeuron** This option quantizes the model by inserting FixNeuron (a combination of QuantizeLinear and DeQuantizeLinear) into the tensor. This is the default value.
* **calibrate_method**:  (Enum) This parameter is used to set the power-of-2 scale quantization method for DPU devices. It currently supports two methods: 'vai_q_onnx.PowerOfTwoMethod.NonOverflow' and 'vai_q_onnx.PowerOfTwoMethod.MinMSE'. The default value is 'vai_q_onnx.PowerOfTwoMethod.MinMSE'.
* **input_nodes**:  (List of Strings) This parameter is a list of the names of the starting nodes to be quantized. Nodes in the model before these nodes will not be quantized. For example, this argument can be used to skip some pre-processing nodes or stop the first node from being quantized. The default value is an empty list ([]).
* **output_nodes**: (List of Strings) This parameter is a list of the names of the end nodes to be quantized. Nodes in the model after these nodes will not be quantized. For example, this argument can be used to skip some post-processing nodes or stop the last node from being quantized. The default value is an empty list ([]).
* **op_types_to_quantize**:  (List of Strings or None) If specified, only operators of the given types will be quantized (e.g., ['Conv'] to only quantize Convolutional layers). By default, all supported operators will be quantized.
* **per_channel**: (Boolean) Determines whether weights should be quantized per channel. For DPU devices, this must be set to False as they currently do not support per-channel quantization.
* **reduce_range**: (Boolean) If True, quantizes weights with 7-bits. For DPU devices, this must be set to False as they currently do not support reduced range quantization.
* **activation_type**: (QuantType) Specifies the quantization data type for activations. For DPU devices, this must be set to QuantType.QInt8. For more details on data type selection, refer to the ONNX Runtime quantization documentation.
* **weight_type**: (QuantType) Specifies the quantization data type for weights. For DPU devices, this must be set to QuantType.QInt8.
* **nodes_to_quantize**:(List of Strings or None) If specified, only the nodes in this list are quantized. The list should contain the names of the nodes, for example, ['Conv__224', 'Conv__252'].
* **nodes_to_exclude**:(List of Strings or None) If specified, the nodes in this list will be excluded from quantization.
* **optimize_model**:(Boolean) If True, optimizes the model before quantization. However, this is not recommended as optimization changes the computation graph, making the debugging of quantization loss difficult.
* **use_external_data_format**:  (Boolean) This option is used for large size (>2GB) model. The default is False.
* **extra_options**:  (Dictionary or None) Contains key-value pairs for various options in different cases. Current used:<br>
                **extra.Sigmoid.nnapi = True/False**  (Default is False)
                <br>**ActivationSymmetric = True/False**: If True, calibration data for activations is symmetrized. The default is False. When using PowerOfTwoMethod for calibration, this should always be set to True.
                <br>**WeightSymmetric = True/False**: If True, calibration data for weights is symmetrized. The default is True. When using PowerOfTwoMethod for calibration, this should always be set to True.
                <br>**EnableSubgraph = True/False**:  If True, the subgraph will be quantized. The default is False. More support for this feature is planned in the future.
                <br>**ForceQuantizeNoInputCheck = True/False**:
                    If True, latent operators such as maxpool and transpose will always quantize their inputs, generating quantized outputs even if their inputs have not been quantized. The default behavior can be overridden for specific nodes using nodes_to_exclude.
                <br>**MatMulConstBOnly = True/False**:
                    If True, only MatMul operations with a constant 'B' will be quantized. The default is False for static mode.
                <br>**AddQDQPairToWeight = True/False**:
                     If True, both QuantizeLinear and DeQuantizeLinear nodes are inserted for weight, maintaining its floating-point format. The default is False, which quantizes floating-point weight and feeds it solely to an inserted DeQuantizeLinear node. In the PowerOfTwoMethod calibration method, QDQ should always appear as a pair, hence this should be set to True.
                <br>**OpTypesToExcludeOutputQuantization = list of op type**:
                     If specified, the output of operators with these types will not be quantized. The default is an empty list.
                <br>**DedicatedQDQPair = True/False**: If True, an identical and dedicated QDQ pair is created for each node. The default is False, allowing multiple nodes to share a single QDQ pair as their inputs.
                <br>**QDQOpTypePerChannelSupportToAxis = dictionary**:
                     Sets the channel axis for specific operator types (e.g., {'MatMul': 1}). This is only effective when per-channel quantization is supported and per_channel is True. If a specific operator type supports per-channel quantization but no channel axis is explicitly specified, the default channel axis will be used. For DPU devices, this must be set to {} as per-channel quantization is currently unsupported.
                <br>**CalibTensorRangeSymmetric = True/False**:
                    If True, the final range of the tensor during calibration will be symmetrically set around the central point "0". The default is False. In PowerOfTwoMethod calibration method, this should always be set to True.
                <br>**CalibMovingAverage = True/False**:
                    If True, the moving average of the minimum and maximum values will be computed when the calibration method selected is MinMax. The default is False. In PowerOfTwoMethod calibration method, this should be set to False.
                <br>**CalibMovingAverageConstant = float**:
                    Specifies the constant smoothing factor to use when computing the moving average of the minimum and maximum values. The default is 0.01. This is only effective when the calibration method selected is MinMax and CalibMovingAverage is set to True. In PowerOfTwoMethod calibration method, this option is unsupported.


## License

Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
