To support multiple quantization configurations, vai_q_pytorch provides quantization configuration file in json format.

### Usage

In order to make the customized configuration take effect, we only need to pass the configuration file to torch_quantizer API. 
```shell
config_file = "./pytorch_quantize_config.json"
quantizer = torch_quantizer(quant_mode=quant_mode, 
                            module=model, 
                            input_args=(input), 
                            device=device, 
                            quant_config_file=config_file)
```

### Configuration List

Quantization configurations consist of two part. <br>
The first part is global quantizer settings, which are listed as follows.
```shell
convert_relu6_to_relu: whether to convert relu6 to relu, options: true, false.
include_cle: whether to use cross layer equalization, options: true, false.
include_bias_corr: whether to use bias correction, options: true, false.
keep_first_last_layer_accuracy: whether to skip quantization of the first and last layers. Reserved word, options: true, false, default: false. Not used currently, . 
keep_add_layer_accuracy: whether to skip quantization of "add" layers. Reserved word, options: true, false, default: False. Not used currently.
target_device: device to deploy quantized model, options: DPU, CPU, GPU.
quantizable_data_type: tensor types to be quantized in model.
```
The second part is the quantization parameters used by the quantizer, which are listed as follows.
```shell
bit_width: bit width used in quantization.
method: method used to calibrate the quantization “scale”, options: maxmin, percentile, entropy, mse, diffs.
round_mode: rounding method in quantization process, options: half_even, half_up, half_down, std_round.
symmetry: whether to use symmetric quantization, options: true, false.
per_channel: whether to use per_channel quantization, options: true, false.
signed: whether to use signed quantization, options: true, false.
narrow_range: whether to use symmetric integer range for signed quantization, options: true, false.
scale_type: scale type used in quantization process, options: float, poweroftwo.
calib_statistic_method: method to choose one optimal “scale” if got different scales using multiple batch data, option: modal, max, mean, median.
```

### Example

There is example code in example/resnet18_quant.py, which could use the file example/quantize_config.json as its configuration file. Run command with "--config_file pytorch_quantize_config.json" to quantize model.
```shell
cd example
python resnet18_quant.py --quant_mode calib --config_file pytorch_quantize_config.json
python resnet18_quant.py --quant_mode test --config_file pytorch_quantize_config.json
```

### Hierarchical Configuration

Quantization configurations is in hierarchical structure. 
- If configuration file is not provided in the torch_quantizer API, the default configuration will be used, which is adapted to DPU device and uses poweroftwo quantization method.
- If configuration file is provided, model configuration, including global quantizer settings and global quantization parameters are required. If only model configuration is provided in the configuration file, all tensors in the model will use the same configuration.
- Layer configuration could be used to set some layers to specific configuration parameters.

#### Default Configurations
Details of default configuration are shown below.
```shell
"convert_relu6_to_relu": false,
"convert_silu_to_hswish": false,
"include_cle": true,
"keep_first_last_layer_accuracy": false,
"keep_add_layer_accuracy": false,
"include_bias_corr": true,
"target_device": "DPU",
"quantizable_data_type": ["input", "weights", "bias", "activation"],
"bit_width": 8, "method": "diffs", "round_mode": "std_round", "symmetry": true, "per_channel": false, "signed": true, "narrow_range": false, "scale_type": "poweroftwo", "calib_statistic_method": "modal"
```

#### Model Configurations
In the example configuration file "example/pytorch_quantize_config.json", the global quantizer settings are set under their respective keywords. And global quantization parameters must be set under the "overall_quantize_config" keyword. As shown below.
```shell
"convert_relu6_to_relu": false,
"convert_silu_to_hswish": false,
  "include_cle": false,
  "keep_first_last_layer_accuracy": false,
  "keep_add_layer_accuracy": false,
  "include_bias_corr": false,
  "target_device": "CPU",
  "quantizable_data_type": [
    "input",
    "weights",
    "bias",
    "activation"
  ],
  "overall_quantize_config": {
    "bit_width": 8,
    "method": "maxmin",
    "round_mode": "half_even",
    "symmetry": true,
    "per_channel": false,
    "signed": true,
    "narrow_range": false,
    "scale_type": "float",
    "calib_statistic_method": "max"
  }
```
Optionally, the quantization configuration of different tensors in the model can be set separately. And the configurations must be set in "tensor_quantize_config" keyword. <br>
And in the example configuration file, we just change the quantization method of activation to "mse". The rest of the parameters are used the same as the global parameters.
```shell
"tensor_quantize_config": {
    "activation": {
      "method": "mse"
    }
  }
```

#### Layer Configurations
Layer quantization configurations must be added in the "layer_quantize_config" list. And two parameter configuration methods, layer type and layer name, are supported. There are five notes to do layer configuration.
- Each individual layer configuration must be in dictionary format.
- In each layer configuration, the "quantizable_data_type" and "overall_quantize_config" parameter are required. And in "overall_quantize_config" parameter, all quantization parameters for this layer need to be included.
- If setting based on layer type, the “layer_name” parameter should be null. 
- If setting based on layer name, the model needs to run the calibration process firstly, then pick the required layer name from the generated .py file in quantized_result directory. Besides, the “layer_type” parameter should be null.
- Same as model configurations, the quantization configuration of different tensors in the layer can be set separately. And they must be set in "tensor_quantize_config" keywords. <br>

In the example configuration file, there are two layer configurations. One is based on layer type, the other is based on layer name.<br>
In the layer configuration based on layer type, torch.nn.Conv2d layer need to set to specific quantization parameters.<br>
And the "per_channel" parameter of weight is set to "true", "method" parameter of activation is set to "entropy". 
```shell
{
  "layer_type": "torch.nn.Conv2d",
  "layer_name": null,
  "quantizable_data_type": [
    "weights",
    "bias",
    "activation"
  ],
  "overall_quantize_config": {
    "bit_width": 8,
    "method": "maxmin",
    "round_mode": "half_even",
    "symmetry": true,
    "per_channel": false,
    "signed": true,
    "narrow_range": false,
    "scale_type": "float",
    "calib_statistic_method": "max"
  },
  "tensor_quantize_config": {
    "weights": {
      "per_channel": true
    },
    "activation": {
      "method": "entropy"
    }
  }
}
```
In the layer configuration based on layer name, the layer named "ResNet::ResNet/Conv2d[conv1]/input.2" need to set to specific quantization parameters.<br>
And the round_mode of activation in this layer is set to "half_up". 
```shell
{
  "layer_type": null,
  "layer_name": "ResNet::ResNet/Conv2d[conv1]/input.2",
  "quantizable_data_type": [
    "weights",
    "bias",
    "activation"
  ],
  "overall_quantize_config": {
    "bit_width": 8,
    "method": "maxmin",
    "round_mode": "half_even",
    "symmetry": true,
    "per_channel": false,
    "signed": true,
    "narrow_range": false,
    "scale_type": "float",
    "calib_statistic_method": "max"
  },
  "tensor_quantize_config": {
    "activation": {
      "round_mode": "half_up"
    }
  }
}
```
The layer name "ResNet::ResNet/Conv2d[conv1]/input.2" is picked from generated file "quantize_result/ResNet.py" of example code "example/resnet18_quant.py". <br>
- Run the example code with command "python resnet18_quant.py --subset_len 100". The quantize_result/ResNet.py file is generated.  
- In the file, the name of first convolution layer is "ResNet::ResNet/Conv2d[conv1]/input.2". 
- Copy the layer name to quantization configuration file if this layer is set to specific configuration.
```shell 
import torch
import pytorch_nndct as py_nndct
class ResNet(torch.nn.Module):
  def __init__(self):
    super(ResNet, self).__init__()
    self.module_0 = py_nndct.nn.Input() #ResNet::input_0
    self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[7, 7], stride=[2, 2], padding=[3, 3], dilation=[1, 1], groups= 1, bias=True) #ResNet::ResNet/Conv2d[conv1]/input.2
```

#### Configuration Restrictions
Due to the restrict of DPU device, if quantized models need to be deployed in DPU device, the quantization configuration should meet the restrictions as below.
```shell
method: diffs or maxmin
round_mode: std_round for weights, bias, and input; half_up for activation.
symmetry: true
per_channel: false
signed: true
narrow_range: true
scale_type: poweroftwo
calib_statistic_method: modal.
```
And for CPU and GPU device, there is no restriction as DPU device. However, there are some conflicts when using different configurations. For example, if calibration method is ‘maxmin’, ‘percentile’, ‘mse’ or ‘entropy’,  the calibration statistic method ‘modal’ is not supported. If symmetry mode is asymmetry, the calibration method ‘mse’ and ‘entropy’ are not supported. Quantization tool will give error message if there exist configuration conflicts.
