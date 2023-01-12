# How To Write Quantize Strategy Configs

## Overview

Quantize strategy is the top-level configurations for the quantize tool. Different devices and user cases need different quantize strategy configurations.
For example, the quantization for DPU using power-of-2 scales, symmetry quantizers and need some special processes to make the simulate bit-level accurate with the hardware results.
For other devices supporting floating point scales will need a different quantize strategy.

We provide 3 types of built-in quantize_strategy now, including:
1. "pof2s": power-of-2 scale quantization, mainly used for DPU now.
2. "fs": float scale quantization, mainly used for devices supporint floating point calculation.
3. "tqt": trained quantization threshold for power-of-2 scale quantization, mainly used for QAT for DPU now.

Moreover, the quantize strategy configurations may need to be customized in some user cases.
For example, the user may need to stop quantizing the first and last conv layers or quantize them to 16 bit. 
To help users do those kinds of modification, we provide APIs for users to override the built-in quantize strategies. 
We also provide some handy configurations to let user set some most-common configurations at one time.
For advanced users who want fully control for the quantize tool and do some experiments, we also provide APIs to accept customized quantize strategies.

## Quantizer Config

The `Quantizer` means the quantize function applied to each object. It consumes a float tensor and output a quantized tensor. 
Please note that the quantization is 'fake', which means that the input is quantized to int and then dequantized to float.
We can apply different quantizer for different object in one quantize strategy. It is configured in the quantize_registry_config part of the quantize strategy config.

We provide 4 types of built-in quantizers now and, including:
1. "Pof2SQuantizer": do power-of-2 scale quantization
```python
{
  "type": "Pof2SQuantizer",
  "params":
  {
    "bit_width": 1-16,
    "data_format": "INT/UINT",
    "method": "No_OF/Min_MSE/Min_KL/Percentile",
    "round_mode": "HALF_TO_EVEN/HALF_UP/HALF_TO_ZERO",
    "symmetry": true,
    "per_channel": false,
    "channel_axis": -1,
    "unsigned": false,
    "narrow_range": false,
    "epsilon": 1e-7,
    "min_quantize_pos": -52,
    "max_quantize_pos": 52
  }
}
```
2. "FSQuantizer": do float scale quantization
```python
{
  "quantizer_type": "FSQuantizer",
  "quantizer_params":
  {
    "bit_width": 1-16,
    "data_format": "INT"/"UINT"/"BFP16"/"FP16",
    "method": "No_OF"/"Min_MSE"/"Min_KL"/"Percentile",
    "round_mode": "HALF_TO_EVEN"/"HALF_UP"/"HALF_TO_ZERO",
    "symmetry": false,
    "per_channel": true,
    "channel_axis": -1,
    "epsilon": 1e-7,
    "unsigned": false,
    "narrow_range": false,
    "use_framework_quantizer": true
  }
}
```
3. "MAFSQuantizer": do float scale quantization with moving average min-max of input data.
```python
{
  "type": "MAFSQuantizer",
  "params":
  {
    "bit_width": 1-16,
    "data_format": "INT"/"UINT",
    "method": "No_OF"/"Min_MSE"/"Min_KL"/"Percentile",
    "round_mode": "HALF_TO_EVEN"/"HALF_UP"/"HALF_TO_ZERO",
    "ema_decay": 0.999,
    "symmetry": true,
    "per_channel": false,
    "channel_axis": -1,
    "unsigned": false,
    "narrow_range": false,
    "epsilon": 1e-7
  }
}
```
3. "TQTQuantizer": do trained quantizatio threshold power-of-2 scale quantization.
```python
{
  "type": "TQTQuantizer",
  "params":
  {
    "bit_width": 1-16,
    "data_format": "INT/UINT",
    "method": "No_OF/Min_MSE/Min_KL/Percentile",
    "round_mode": "HALF_TO_EVEN/HALF_UP/HALF_TO_ZERO",
    "symmetry": true,
    "per_channel": false,
    "channel_axis": -1,
    "unsigned": false,
    "narrow_range": false,
    "epsilon": 1e-7
  }
}
```

## Quantize Tool Workflow

The quantize tool workflow mainly have 3 steps:

#### 1. Optimize the float model
Do optimization for the float model, Including
1) Remove the unused model.
2) Fold BN into previous Conv/Dense layers.
3) Convert un-foldable BN into DepthwiseConv layers.
4) Convert TFOpLayer into equivalent keras layers.
5) Convert ReLU6 to ReLU.
6) Do cross-layer-equalization.
7) In QAT mode, do fake folding for Conv-BN.

#### 2. Quantzie the float model
Convert the float model into a quantized model, insert quantizers into the right positions to apply fake quantization. 
Mainly including:
1) For DPU devices, convert layers to DPU version to simulate the behavior, e.g. average_pooling2d, leaky_relu, hard_sigmoid and so on.
2) Insert quantizers to the quantizable patterns.
3) In PTQ mode, will do calibration and generate the quantized model.
4) In QAT mode, will generate the quantized model for QAT, need user codes to do the finetuning.

#### 3. Refine the quantized model
Refine the quantized model to make it deployable. 
1) In QAT mode, do real folding for Conv-BN after finetuning.
2) For DPU devicis, adjust the quantize positions according to the compiler constraints.
3) Do fast-finetuning to adjust the weights and biases.
4) Do bias correction to adjust the biases.
5) Convert the quantizers to framework quantizers.
6) Generate the final quanized model, may need to convert to onnx format.

## Quantize Strategy
It configures all the workflows and also configures how to insert the quantizers.

1. optimize_pipeline_config: configures the optimize pipeline.
2. quantize_pipeline_config: configures the quantize pipeline.
3. refine_pipeline_config: configures the refine pipeline.
4. quantize_registry_config: configures how to quantize the model.

## Quantize Registry Config
Configures how to quantize the model, where and how to insert the quantizers.

#### 1. user_defined_quantize_config
#### 2. input_quantize_config
#### 3. layer_quantize_config
Format:

```python
{
    "pattern_name": "", # String, unique key for this pattern.
    "layer_type": "", # String, layer_type supports "|" to match multiple types, "" to match any types.
    "layer_name": "", # String, layer_name supports regex, used to set fine-grained quantize config.
    "layer_params": {}, # Constraints for the layer parameters, e.g. kernel_size, dilation.
    "layer_inputs": [] # List of layer_quantize_config, recursive configurations for layer inputs, support pattern nesting, must keep order.
        { "layer_type": "xxxx"} # support layer nesting.
        { "pattern": "xxxx"}, # support pattern nesting.
    
    
    "quant_config": [], # List of quant_configs, config how to quantize this input layer.
        {
            "target_name": "", # String, unique key to the quantize target.
            "target_type": "", # String, type of the quantize target, must be one of ["weight", "bias", "output"].
            # target" is the object to insert quantizer op after.
            # "target_name" is the unique key for this layer, "target_type" is the category of this target.
            # Now supports 3 kinds of target_types:
            # 1. "weight", local var of this layer, can be get via get_attr(layer, "target_name").
            # 2. "bias", local var of this layer, can be get via get_attr(layer, "target_name"). 
            #     Spliting local vars into "weight" and "bias" helps us to handle them more efficiently. e.g. user can switch
            #     all layers' bias into 32 bit quantize in user_quantize_config by setting "bias_bit=32" while keeping
            #     all layers' weight unchanged.
            # 3. "output", layer's final output tensor, quantizer will be called after layer.call().

            "shared_quantizer": "", # String, name of another defined quantizer in this pattern.
            "quantizer_config": , # Dict of quantizer config.
            {
                "name": "", # String, unique key in this quantizer.
                "type": "", # String, quantizer type of built-in quantizers or customized quantizer. Available built-in quantizers are ['Pof2SQuantizer', 'FSQuantizer', 'MAFSQuantizer', 'TQTQuantizer'].
                "params": {} # Dict, quantizer parameters.
            }
        }
}
```
