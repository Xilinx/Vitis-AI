## vai_q_pytorch

### Introduction
vai_q_pytorch is short for Vitis AI Quantizer for Pytorch. It is a tool for neural network model optimization with Pytorch model input.
vai_q_pytorch is designed as a part of a standard platform for neural network deep compression. Base on this architecture, working stages of vai_q_pytorch follows:<br>
    1.  Parse neural network computation graph from Pytorch framework to Intermediate Representation (IR).<br>
    2.  Modify the computation graph based on IR. Pruning, quantization and graph optimization are handled in this stage.<br>
    3.  Deploy the modified IR on different platform, such as DPU and Xilinx AI Engine.<br>
    4.  Assemble the modified computation graph back to Pytorch. In that way abilities and tools in Pytorch such as pre-processing, post processing and distribution system can be used.<br>

### Supported

Python version 3.6 ~ 3.7 

Pytorch version 1.1 ~ 1.4 

Classification Models in Torchvision, Pytorch models in Xilinx Modelzoo 

### Install

Installation with Anaconda is suggested. 

To install vai_q_pytorch, do as follows:

#### Pre step 1 : install Pytorch(1.1 or above) and torchvision
    pip install torch==1.1.0 torchvision==0.3.0 
    export CUDA_HOME with your cuda include folder in .bashrc 

#### Pre step 2 : install other dependencies
    pip install -r requirements.txt 

#### Now install the main component:
    cd ./pytorch_binding 
    python setup.py install (for user) 
    python setup.py develop (for developer) 

To create deployed model, XIR library needs to be installed. If just run quantization and check the accuracy, this is not must. 
Refer to Vitis AI document for more information on deployment.

### Tool Usage

This chapter introduce using execution tools and APIs to implement quantization and generated model to be deployed on target hardware.  The APIs are in module nndct/pytorch_binding/pytorch_nndct/apis/quant_api.py:
#### Function torch_quantizer will create a quantizer.
```py
    def torch_quantizer(quant_mode,
                        module,
                        input_args,
                        state_dict_file,
                        output_dir,
                        bitwidth_w,
                        bitwidth_a)
```
    quant_mode: An integer that indicates which quantization mode the process is using. 0 for turning off quantization. 1 for calibration of quantization. 2 for evaluation of quantized model.
    Module: Float module to be quantized.
    Input_args: input tensor with the same shape as real input of float module to be quantized, but the values can be random number.
    State_dict_file: Float module pretrained parameters file. If float module has read parameters in, the parameter is not needed to be set.
    Output_dir: Directory for quantization result and intermediate files. Default is “quantize_result”.
    Bitwidth_w: Global weights and bias quantization bit width. Default is 8.
    Bitwidth_a: Global activation quantization bit width. Default is 8.
#### Function dump_xmodel will create deployed model. 
```py
    def dump_xmodel(output_dir, deploy_check)
```
    Output_dir: Directory for quantizapyttion result and intermediate files. Default is “quantize_result”
    Depoly_check: Flags to control dump of data for accuracy check. Default is False.

For quantization, the original preprocess function, forward function and post-process function will be used for calibration and quantized model, just need to replaced the input module of forward function with converted module (quantizer.quant_model in code). The calibration iteration number and quantized model evaluation iteration number can be controlled as the iteration number in the data loading part code as original model training, validation.
### Quick Start
#### Request of the Pytorch model to be quantized:
1.  Refine the float module which need to be quantized, the refined module should include the forward method only. All post process method defined in float module will be removed in converted module. <br>
2.  Set the refined float module to evaluation status, then use “torch.jit.trace” function to test the float model. Make sure the float module can pass the trace test. <br>

#### An example case is nndct/example/resnet18_quant.py. 

###### Some preparation to run the model:
1.  Install torch vision module <br>
2.  Get ImageNet data set <br>
3.  Download pretrained torch vision resnet-18 model .pth file <br>

###### To call vai_q_pytorch, some parts of code needs to be added:

1.  Import vai_q_pytorch modules <br>

```py
    from pytorch_nndct.apis import torch_quantizer, dump_xmodel
```

2.  Generate a quantizer with quantization needed input and get converted model. <br>

```py
    input = torch.randn([batch_size, 3, 224, 224])
    quantizer = torch_quantizer(
            quant_mode, model, (input))
        quant_model = quantizer.quant_model
```

3.  Forwarding with converted model. <br>

4.  Output quantization result and deploy model. <br>

```py
    quantizer.export_quant_config()
    dump_xmodel()
```

###### Command line and run results

To do calibration of quantization, run the following command line:

    python resnet18_quant.py --quant_mode 1

After it is finished, two important files will be generated under output directory “./quantize_result”. 

    ResNet.py: converted vai_q_pytorch format model, 
    Quant_info.json: quantization steps of tensors got. (Keep it for evaluation of quantized model)
    ResNet_int.xmodel: deployed model

To do evaluation of quantized model, run the following command line:

    python resnet18_quant.py --quant_mode 2

